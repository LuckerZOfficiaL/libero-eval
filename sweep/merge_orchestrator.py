#!/usr/bin/env python
"""Multi-LoRA MERGE evaluation orchestrator (EXP-3).

A merge *job* = {name, method, params, adapters:[task indices], eval_scenes:[LIBERO scenes]}.

Scheduling: up to `n_workers` jobs run concurrently — 2 servers per P6000, so 4 in parallel; any
extra jobs wait in a shared QUEUE and start as a slot frees up. Each worker pulls a job, builds its
merged checkpoint (scripts/merge_loras.py) in /dev/shm, serves it, evaluates ALL of that job's scenes
on its one server (main.py --args.task-name, N trials), records each, tears the server down, then
pulls the next queued job. Concurrent merge builds are capped (RAM) by a separate semaphore.

Resumable: (job, scene) pairs already recorded 'ok' are skipped; a job with no un-done scenes is not
queued. Results append to results/merge/<job>.jsonl (one file per pair+method).

Usage: python sweep/merge_orchestrator.py <plan.json>
"""
import atexit, glob, json, os, queue, re, signal, socket, subprocess, sys, threading, time
from pathlib import Path

REPO = "/home/lzhou/libero-eval"
LORA_ROOT = "/media/pinas/mehdi/libero90_lora"
PLAN = sys.argv[1] if len(sys.argv) > 1 else f"{REPO}/sweep/merge_plan.json"
plan = json.load(open(PLAN))
JOBS = plan["jobs"]
BASE, NORM, NTRIALS, NW = plan["base_model"], plan["norm_stats"], plan["n_trials"], plan["n_workers"]
SAVE_VIDEOS = str(plan.get("save_videos", "all"))   # 'all' | '0'/'none' | k (last k) — passed to main.py
GPUS, PORTS = plan["gpu_per_worker"], plan["port_per_worker"]
RESULTS_DIR = plan.get("results_dir", f"{REPO}/results/merge")   # one file per job: <job>.jsonl
VIDEOS_DIR = plan.get("videos_dir", f"{REPO}/results/merge_videos")
LOGDIR = plan.get("logdir", f"{REPO}/results/merge_logs")
VENV_PY = f"{REPO}/.venv/bin/python"
SHM = "/dev/shm/mergesweep"
CACHE = "/home/lzhou/.cache/uv"
os.makedirs(LOGDIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

def adapter_file(idx):
    return f"{LORA_ROOT}/task{idx}/task{idx}_adapter.safetensors"

# resumable: scan all per-job result files
done = set()
for rf in glob.glob(f"{RESULTS_DIR}/*.jsonl"):
    for line in open(rf):
        try:
            r = json.loads(line)
            if r.get("status") == "ok":
                done.add((r["job"], r["scene"]))
        except Exception:
            pass

results_lock = threading.Lock()
def record(rec):
    rec["ts"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with results_lock:
        with open(f"{RESULTS_DIR}/{rec['job']}.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")

heavy_sem = threading.BoundedSemaphore(1)   # serialize merge builds (each ~21 GB RAM); evals not capped
servers = {}
_plock = threading.Lock()
def log(w, m):
    with _plock:
        print(f"[w{w} {time.strftime('%H:%M:%S')}] {m}", flush=True)

@atexit.register
def _cleanup():
    for p in list(servers.values()):
        try: p.terminate()
        except Exception: pass
def _sig(*_a):
    _cleanup(); os._exit(1)
signal.signal(signal.SIGTERM, _sig); signal.signal(signal.SIGINT, _sig)

def port_open(port):
    with socket.socket() as s:
        s.settimeout(1); return s.connect_ex(("127.0.0.1", port)) == 0

def build_merge(job, worker):
    out = f"{SHM}/w{worker}/merged"
    subprocess.run(["rm", "-rf", out], check=False)
    os.makedirs(out, exist_ok=True)
    adapters = [adapter_file(i) for i in job["adapters"]]
    cmd = [VENV_PY, "scripts/merge_loras.py", "--base-model-dir", BASE, "--adapters", *adapters,
           "--method", job["method"], "--params", json.dumps(job.get("params", {})),
           "--output-dir", out, "--norm-stats", NORM]
    r = subprocess.run(cmd, cwd=REPO, env=dict(os.environ, UV_CACHE_DIR=CACHE), capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"merge build failed: {(r.stderr or r.stdout)[-600:]}")
    return out

def start_server(mergedir, gpu, port, worker):
    env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=str(gpu),
               TORCHDYNAMO_DISABLE="1", UV_CACHE_DIR=CACHE)
    logf = open(f"{LOGDIR}/server_w{worker}.log", "a")
    p = subprocess.Popen([VENV_PY, "scripts/serve_policy.py", "--port", str(port),
                          "policy:checkpoint", "--policy.config", "pi05_libero", "--policy.dir", mergedir],
                         cwd=REPO, env=env, stdout=logf, stderr=subprocess.STDOUT)
    for _ in range(300):
        if port_open(port): return p
        if p.poll() is not None: raise RuntimeError("server died during load")
        time.sleep(3)
    p.terminate(); raise RuntimeError("server did not listen in time")

def run_eval(scene, gpu, port, worker, job_name):
    env = dict(os.environ, MUJOCO_GL="egl", MUJOCO_EGL_DEVICE_ID=str(gpu), UV_CACHE_DIR=CACHE)
    vids = f"{VIDEOS_DIR}/{job_name}/{scene}"
    logf = f"{LOGDIR}/eval_w{worker}.log"
    with open(logf, "a") as lf:
        lf.write(f"\n===== job={job_name} scene={scene} =====\n"); lf.flush()
        subprocess.run([VENV_PY, "examples/libero/main.py", "--args.task-suite-name", "libero_90",
                        "--args.task-name", scene, "--args.num-trials-per-task", str(NTRIALS),
                        f"--args.save-videos={SAVE_VIDEOS}",   # = form: safe for negative values (last-k)
                        "--args.port", str(port), "--args.video-out-path", vids],
                       cwd=REPO, env=env, stdout=lf, stderr=subprocess.STDOUT)
    text = open(logf).read().split(f"===== job={job_name} scene={scene} =====")[-1]
    rates = re.findall(r"Total success rate: ([0-9.]+)", text)
    eps = re.findall(r"Total episodes: (\d+)", text)
    return (float(rates[-1]), int(eps[-1])) if rates and eps else (None, None)

def worker_fn(worker, job_q):
    """Pull jobs from the shared queue until it's empty. Each job: build -> serve -> eval its scenes."""
    gpu, port = GPUS[worker], PORTS[worker]
    time.sleep(worker * 3)  # small stagger so builds don't all fire at once
    while True:
        try:
            job = job_q.get_nowait()
        except queue.Empty:
            break
        scenes = [s for s in job["eval_scenes"] if (job["name"], s) not in done]
        if not scenes:
            continue
        server = None
        try:
            with heavy_sem:  # bound concurrent RAM-heavy builds
                log(worker, f"{job['name']}: merging {job['adapters']} ({job['method']}) [{job_q.qsize()} queued]")
                md = build_merge(job, worker)
                log(worker, f"{job['name']}: serving gpu={gpu} port={port}")
                server = start_server(md, gpu, port, worker); servers[worker] = server
                subprocess.run(["rm", "-rf", md], check=False)   # weights on GPU now
            for scene in scenes:
                t0 = time.time()
                rate, neps = run_eval(scene, gpu, port, worker, job["name"])
                dt = round(time.time() - t0)
                if server.poll() is not None:
                    log(worker, f"{job['name']}/{scene}: SERVER DIED")
                    record({"job": job["name"], "scene": scene, "status": "server_died"})
                    break
                record({"job": job["name"], "method": job["method"], "params": job.get("params", {}),
                        "adapters": job["adapters"], "scene": scene, "n_trials": neps,
                        "success_rate": rate, "n_success": None if rate is None else round(rate * neps),
                        "status": "ok" if rate is not None else "parse_fail", "secs": dt, "worker": worker})
                log(worker, f"{job['name']}/{scene} -> rate={rate} n={neps} ({dt}s)")
        except Exception as ex:
            log(worker, f"{job['name']}: ERROR {ex}")
            record({"job": job["name"], "scenes": scenes, "status": "error", "error": str(ex)[:400]})
        finally:
            if server is not None:
                try: server.terminate(); server.wait(timeout=20)
                except Exception: server.kill()
                servers.pop(worker, None); time.sleep(2)
    log(worker, "WORKER DONE")

if __name__ == "__main__":
    job_q = queue.Queue()
    for job in JOBS:
        if any((job["name"], s) not in done for s in job["eval_scenes"]):
            job_q.put(job)
    log("main", f"plan={PLAN} jobs={job_q.qsize()}/{len(JOBS)} to-do, up to {NW} in parallel (2/GPU), "
                f"trials={NTRIALS}, already_done={len(done)}")
    threads = [threading.Thread(target=worker_fn, args=(w, job_q)) for w in range(NW)]
    for t in threads: t.start()
    for t in threads: t.join()
    log("main", "ALL WORKERS DONE")
