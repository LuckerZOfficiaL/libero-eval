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
import atexit, glob, json, os, pathlib, queue, re, signal, socket, subprocess, sys, threading, time
from pathlib import Path

REPO = "/home/lzhou/libero-eval"
LORA_ROOT = "/media/pinas/mehdi/libero90_lora"
PLAN = sys.argv[1] if len(sys.argv) > 1 else f"{REPO}/sweep/merge_plan.json"
plan = json.load(open(PLAN))
JOBS = plan["jobs"]
BASE, NORM, NTRIALS, NW = plan["base_model"], plan["norm_stats"], plan["n_trials"], plan["n_workers"]
SAVE_VIDEOS = str(plan.get("save_videos", "all"))   # 'all' | '0'/'none' | k (last k) — passed to main.py
GPUS, PORTS = plan["gpu_per_worker"], plan["port_per_worker"]
# 10-21 (shiva GPU 1 failed): optional maps from the plan's GPU index to the CUDA device (e.g. a UUID) and to the EGL device
# used by the simulator; "0=GPU-...,2=GPU-...". Unset -> previous behaviour (the index for both).
_MAP = lambda var: dict(kv.split("=", 1) for kv in os.environ.get(var, "").split(",") if "=" in kv)
CUDA_MAP, EGL_MAP = _MAP("ORCH_CUDA_MAP"), _MAP("ORCH_EGL_MAP")
RESULTS_DIR = plan.get("results_dir", f"{REPO}/results/merge")   # one file per job: <job>.jsonl
# Which evaluation entry point to run. Default is the live one; plans may point at the instrumented
# copy (main_dense.py) to also record per-episode PROGRESS and SUCCESS_STEP. Kept as a plan key so an
# instrumented run cannot disturb a concurrent standard one.
EXTRA = {}                                    # (job, scene) -> per-episode PROGRESS / SUCCESS_STEP
EVAL_SCRIPT = plan.get("eval_script", "examples/libero/main.py")
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
    # per-episode PROGRESS / SUCCESS_STEP, present only when the plan selected the instrumented
    # eval script; harmless (and absent) for every standard run.
    ex = EXTRA.pop((rec.get("job"), rec.get("scene")), None)
    if ex:
        rec.update(ex)
    with results_lock:
        with open(f"{RESULTS_DIR}/{rec['job']}.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")

# serialize merge builds + server start (each build ~21 GB RAM); evals not capped. The whole block takes ~65 s,
# which caps a 7-worker host at ~55 job starts/hour on short evals -> plan key "heavy_parallel" (default 1)
heavy_sem = threading.BoundedSemaphore(int(plan.get("heavy_parallel", 1)))
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

def job_adapters(job):
    """Adapter files for a job: explicit paths (external adapters) or fleet indices."""
    return list(job["adapter_paths"]) if job.get("adapter_paths") else [adapter_file(i) for i in job["adapters"]]

def job_sig(job):
    """Merge signature: jobs with identical (adapters, method, params, norm) share one built model."""
    import hashlib
    adapters = job_adapters(job) + [job.get("norm_stats", "")]
    return hashlib.md5(json.dumps([adapters, job["method"], job.get("params", {})]).encode()).hexdigest()[:16]

_SIG_LOCKS, _SIG_GUARD = {}, threading.Lock()
def _sig_lock(sig):
    # 10-07: two workers with the same merge signature raced (the second rm -rf'd the first's in-progress build:
    # 'Directory not empty' errors in EXP-108/109). One lock per signature: the second waits, then reuses.
    with _SIG_GUARD:
        return _SIG_LOCKS.setdefault(sig, threading.Lock())

def build_merge(job, worker):
    # EXP-101: a job may point at a checkpoint the driver already wrote (dense merges that are not low-rank
    # adapters); serve it as-is. The driver owns and deletes it.
    if job.get("prebuilt_dir"):
        log(worker, f"{job['name']}: serving prebuilt {job['prebuilt_dir']}")
        return job["prebuilt_dir"]
    # Cache by merge signature — e.g. several probe jobs splitting one soup's eval scenes build
    # ONCE. A cached model is evicted when the last pending job that needs it finishes (refcount
    # in SIG_REMAINING), so heterogeneous plans don't accumulate models in tmpfs.
    sig = job_sig(job)
    with _sig_lock(sig):
        return _build_locked(job, worker, sig)

def _build_locked(job, worker, sig):
    adapters = job_adapters(job)
    out = f"{SHM}/cache/{sig}"
    if os.path.exists(f"{out}/model.safetensors"):
        log(worker, f"{job['name']}: reusing cached build {sig}")
        return out
    tmp = f"{out}.build"
    subprocess.run(["rm", "-rf", tmp, out], check=False)
    os.makedirs(tmp, exist_ok=True)
    cmd = [VENV_PY, "scripts/merge_loras.py", "--base-model-dir", BASE, "--adapters", *adapters,
           "--method", job["method"], "--params", json.dumps(job.get("params", {})),
           "--output-dir", tmp, "--norm-stats", job.get("norm_stats", NORM)]
    r = subprocess.run(cmd, cwd=REPO, env=dict(os.environ, UV_CACHE_DIR=CACHE), capture_output=True, text=True)
    if r.returncode != 0:
        subprocess.run(["rm", "-rf", tmp], check=False)
        raise RuntimeError(f"merge build failed (rc={r.returncode}): {(r.stderr or r.stdout)[-600:]}")
    os.rename(tmp, out)
    return out

def start_server(mergedir, gpu, port, worker):
    env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=CUDA_MAP.get(str(gpu), str(gpu)),
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

def run_eval(scene, gpu, port, worker, job_name, suite="libero_90"):
    env = dict(os.environ, MUJOCO_GL="egl", MUJOCO_EGL_DEVICE_ID=EGL_MAP.get(str(gpu), str(gpu)), UV_CACHE_DIR=CACHE)
    if EGL_MAP or CUDA_MAP: env.pop("CUDA_VISIBLE_DEVICES", None)   # the client needs no CUDA; robosuite asserts EGL id in it
    # 10-26 (shiva GPU 1 off the bus: every EGL client hangs in the NVIDIA modeset driver): ORCH_GL=osmesa renders on the CPU
    if os.environ.get("ORCH_GL", "egl") == "osmesa":
        env.update(MUJOCO_GL="osmesa", PYOPENGL_PLATFORM="osmesa", LP_NUM_THREADS=os.environ.get("LP_NUM_THREADS", "4"))
        env.pop("MUJOCO_EGL_DEVICE_ID", None)
    vids = f"{VIDEOS_DIR}/{job_name}/{scene}"
    logf = f"{LOGDIR}/eval_w{worker}_p{port}.log"   # 11-05: per port; concurrent orchestrators interleaved eval_w{worker}.log
    with open(logf, "a") as lf:
        lf.write(f"\n===== job={job_name} scene={scene} =====\n"); lf.flush()
        extra = ["--args.state-offset", str(plan["state_offset"])] if plan.get("state_offset") else []
        if plan.get("ood_shift"): extra += ["--args.ood-shift", str(plan["ood_shift"])]
        if plan.get("dump_obs_dir"): extra += ["--args.dump-obs-path", f"{plan['dump_obs_dir']}/{job_name}/{scene}"]
        if plan.get("dump_stride"): extra += ["--args.dump-stride", str(plan["dump_stride"])]
        if plan.get("state_pool"): extra += ["--args.state-pool", str(plan["state_pool"])]
        subprocess.run([VENV_PY, EVAL_SCRIPT, "--args.task-suite-name", suite, *extra,
                        "--args.task-name", scene, "--args.num-trials-per-task", str(NTRIALS),
                        f"--args.save-videos={SAVE_VIDEOS}",   # = form: safe for negative values (last-k)
                        "--args.port", str(port), "--args.video-out-path", vids],
                       cwd=REPO, env=env, stdout=lf, stderr=subprocess.STDOUT)
    text = open(logf).read().split(f"===== job={job_name} scene={scene} =====")[-1]
    rates = re.findall(r"Total success rate: ([0-9.]+)", text)
    eps = re.findall(r"Total episodes: (\d+)", text)
    prog = [float(x) for x in re.findall(r"PROGRESS: ([0-9.]+)", text)]
    sstep = [int(x) for x in re.findall(r"SUCCESS_STEP: (-?\d+)", text)]
    if prog or sstep:
        EXTRA[(job_name, scene)] = {"progress": prog, "success_step": sstep}
    nerr = text.count("Caught exception")   # 11-05: a server error (e.g. CUDA OOM) ends an episode as a FAILURE in the client
    if nerr: return (None, -nerr)
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
                log(worker, f"{job['name']}: merging {job.get('adapters') or [pathlib.Path(a).parent.parent.name for a in job_adapters(job)]} ({job['method']}) [{job_q.qsize()} queued]")
                md = build_merge(job, worker)
                log(worker, f"{job['name']}: serving gpu={gpu} port={port}")
                server = start_server(md, gpu, port, worker); servers[worker] = server
                # NOTE: cached build dirs are kept (other jobs may share them); cleared in main at end.
            for scene in scenes:
                t0 = time.time()
                rate, neps = run_eval(scene, gpu, port, worker, job["name"], job.get("suite", "libero_90"))
                dt = round(time.time() - t0)
                if server.poll() is not None:
                    log(worker, f"{job['name']}/{scene}: SERVER DIED")
                    record({"job": job["name"], "scene": scene, "status": "server_died"})
                    break
                record({"job": job["name"], "method": job["method"], "params": job.get("params", {}),
                        "adapters": job["adapters"], "scene": scene, "n_trials": neps,
                        "success_rate": rate, "n_success": None if rate is None else round(rate * neps),
                        "status": "ok" if rate is not None else ("episode_error" if neps and neps < 0 else "parse_fail"), "secs": dt, "worker": worker})
                log(worker, f"{job['name']}/{scene} -> rate={rate} n={neps} ({dt}s)")
        except Exception as ex:
            log(worker, f"{job['name']}: ERROR {ex}")
            record({"job": job["name"], "scenes": scenes, "status": "error", "error": str(ex)[:400]})
        finally:
            sig = job_sig(job)
            with sig_lock:
                SIG_REMAINING[sig] -= 1
                if SIG_REMAINING[sig] <= 0:
                    subprocess.run(["rm", "-rf", f"{SHM}/cache/{sig}"], check=False)
            if server is not None:
                try: server.terminate(); server.wait(timeout=20)
                except Exception: server.kill()
                servers.pop(worker, None); time.sleep(2)
    log(worker, "WORKER DONE")

sig_lock = threading.Lock()
SIG_REMAINING: dict = {}

if __name__ == "__main__":
    subprocess.run(["rm", "-rf", f"{SHM}/cache"], check=False)   # stale cached builds from prior runs
    job_q = queue.Queue()
    import collections
    pending = [job for job in JOBS if any((job["name"], s) not in done for s in job["eval_scenes"])]
    SIG_REMAINING.update(collections.Counter(job_sig(job) for job in pending))
    for job in pending:
        job_q.put(job)
    log("main", f"plan={PLAN} jobs={job_q.qsize()}/{len(JOBS)} to-do, up to {NW} in parallel (2/GPU), "
                f"trials={NTRIALS}, already_done={len(done)}")
    threads = [threading.Thread(target=worker_fn, args=(w, job_q)) for w in range(NW)]
    for t in threads: t.start()
    for t in threads: t.join()
    subprocess.run(["rm", "-rf", f"{SHM}/cache"], check=False)
    log("main", "ALL WORKERS DONE")
