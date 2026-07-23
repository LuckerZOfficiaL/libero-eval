#!/usr/bin/env python
"""Baseline sweep orchestrator: evaluate each single-task LoRA adapter on its own LIBERO scene(s).

For each adapter (task_index): build base+adapter merge in /dev/shm -> serve on a dedicated
GPU/port -> delete the merge (weights are on GPU) -> eval each of the adapter's scenes with
--args.task-name (single scene only, N trials) -> record success rate -> next adapter.

Runs `n_workers` worker threads in parallel; each owns one GPU+port and a slice of adapters.
Resumable: (task_index, scene) pairs already recorded 'ok' are skipped. Results are appended
incrementally to results/baseline_results.jsonl so partial progress always survives.

Usage: python sweep/orchestrator.py [plan.json]
"""
import json, os, re, socket, subprocess, sys, threading, time, atexit, pathlib

REPO = "/home/lzhou/libero-eval"
PLAN = sys.argv[1] if len(sys.argv) > 1 else f"{REPO}/sweep/plan.json"
plan = json.load(open(PLAN))
adapters = {a["task_index"]: a for a in plan["adapters"]}
NW = plan["n_workers"]
BASE, NORM, NTRIALS = plan["base_model"], plan["norm_stats"], plan["n_trials"]
RESULTS = plan.get("results_file", f"{REPO}/results/baseline_results.jsonl")
VIDEOS_DIR = plan.get("videos_dir", f"{REPO}/results/videos")
LOGDIR = plan.get("logdir", f"{REPO}/results/logs")
VENV_PY = f"{REPO}/.venv/bin/python"
SHM = "/dev/shm/sweep"
CACHE = "/home/lzhou/.cache/uv"
os.makedirs(LOGDIR, exist_ok=True)
os.makedirs(f"{SHM}/stage", exist_ok=True)

# --- resumable: collect already-completed (task_index, scene) ---
done = set()
if os.path.exists(RESULTS):
    for line in open(RESULTS):
        try:
            r = json.loads(line)
            if r.get("status") == "ok":
                done.add((r["task_index"], r["scene"]))
        except Exception:
            pass

results_lock = threading.Lock()
def record(rec):
    rec["ts"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with results_lock:
        with open(RESULTS, "a") as f:
            f.write(json.dumps(rec) + "\n")

# limit concurrent RAM-heavy phases (merge build + server load) across all workers
heavy_sem = threading.BoundedSemaphore(2)
servers = {}          # worker -> Popen (for cleanup)
_print_lock = threading.Lock()
def log(worker, msg):
    with _print_lock:
        print(f"[w{worker} {time.strftime('%H:%M:%S')}] {msg}", flush=True)

@atexit.register
def _cleanup():
    for p in list(servers.values()):
        try:
            p.terminate()
        except Exception:
            pass

import signal
def _sig(*_a):
    _cleanup()
    os._exit(1)
signal.signal(signal.SIGTERM, _sig)
signal.signal(signal.SIGINT, _sig)

def port_open(port):
    with socket.socket() as s:
        s.settimeout(1)
        return s.connect_ex(("127.0.0.1", port)) == 0

def stage_adapter(idx):
    d = f"{SHM}/stage/task{idx}"
    os.makedirs(d, exist_ok=True)
    a = adapters[idx]
    for src, name in [(a["adapter_file"], "adapter_model.safetensors"),
                      (a["adapter_config"], "adapter_config.json")]:
        link = f"{d}/{name}"
        if os.path.lexists(link):
            os.remove(link)
        os.symlink(src, link)
    return d

def build_merge(idx, worker):
    stage = stage_adapter(idx)
    out = f"{SHM}/w{worker}/merged"
    subprocess.run(["rm", "-rf", out], check=False)
    os.makedirs(out, exist_ok=True)
    env = dict(os.environ, UV_CACHE_DIR=CACHE)
    r = subprocess.run(
        [VENV_PY, "scripts/apply_lora.py", "--base-model-dir", BASE,
         "--lora-checkpoint-dir", stage, "--output-dir", out],
        cwd=REPO, env=env, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"merge failed: {(r.stderr or r.stdout)[-600:]}")
    ns = f"{out}/assets/physical-intelligence/libero"
    os.makedirs(ns, exist_ok=True)
    subprocess.run(["cp", NORM, f"{ns}/norm_stats.json"], check=True)
    return out

def start_server(mergedir, gpu, port, worker):
    env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=str(gpu),
               TORCHDYNAMO_DISABLE="1", UV_CACHE_DIR=CACHE)
    logf = open(f"{LOGDIR}/server_w{worker}.log", "a")
    p = subprocess.Popen(
        [VENV_PY, "scripts/serve_policy.py", "--port", str(port), "policy:checkpoint",
         "--policy.config", "pi05_libero", "--policy.dir", mergedir],
        cwd=REPO, env=env, stdout=logf, stderr=subprocess.STDOUT)
    for _ in range(300):  # up to ~15 min
        if port_open(port):
            return p
        if p.poll() is not None:
            raise RuntimeError("server process died during load")
        time.sleep(3)
    p.terminate()
    raise RuntimeError("server did not start listening in time")

def run_eval(scene, port, worker, idx, gpu):
    # Render on the worker's P6000 (not the tiny K620): EGL device index == nvidia-smi index here.
    env = dict(os.environ, MUJOCO_GL="egl", MUJOCO_EGL_DEVICE_ID=str(gpu), UV_CACHE_DIR=CACHE)
    vids = f"{VIDEOS_DIR}/task{idx}/{scene}"  # per-scene subfolder so multi-scene adapters don't collide
    logf = f"{LOGDIR}/eval_w{worker}_task{idx}.log"
    with open(logf, "a") as lf:
        lf.write(f"\n===== task{idx} scene={scene} =====\n"); lf.flush()
        r = subprocess.run(
            [VENV_PY, "examples/libero/main.py",
             "--args.task-suite-name", "libero_90", "--args.task-name", scene,
             "--args.num-trials-per-task", str(NTRIALS), "--args.port", str(port),
             "--args.video-out-path", vids],
            cwd=REPO, env=env, stdout=lf, stderr=subprocess.STDOUT)
    text = open(logf).read()
    rates = re.findall(r"Total success rate: ([0-9.]+)", text)
    eps = re.findall(r"Total episodes: (\d+)", text)
    if rates and eps:
        return float(rates[-1]), int(eps[-1]), r.returncode
    return None, None, r.returncode

def worker_fn(worker):
    gpu = plan["gpu_per_worker"][worker]
    port = plan["port_per_worker"][worker]
    my_adapters = plan["worker_adapters"][worker]
    time.sleep(worker * 15)  # stagger startups
    for idx in my_adapters:
        a = adapters[idx]
        scenes = [s for s in a["scenes"] if (idx, s) not in done]
        if not scenes:
            log(worker, f"task{idx}: all scenes already done, skipping")
            continue
        server = None
        try:
            with heavy_sem:  # bound concurrent merge+load (RAM)
                log(worker, f"task{idx}: building merge for {len(scenes)} scene(s)")
                mergedir = build_merge(idx, worker)
                log(worker, f"task{idx}: starting server gpu={gpu} port={port}")
                server = start_server(mergedir, gpu, port, worker)
                servers[worker] = server
                subprocess.run(["rm", "-rf", mergedir], check=False)  # weights on GPU now
            for scene in scenes:
                t0 = time.time()
                rate, neps, rc = run_eval(scene, port, worker, idx, gpu)
                dt = round(time.time() - t0)
                if server.poll() is not None:
                    log(worker, f"task{idx}: SERVER DIED during {scene}")
                    record({"task_index": idx, "scene": scene, "status": "server_died", "worker": worker})
                    break
                status = "ok" if rate is not None else "parse_fail"
                record({"task_index": idx, "language": a["language"], "scene": scene,
                        "n_trials": neps, "success_rate": rate,
                        "n_success": None if rate is None else round(rate * neps),
                        "status": status, "secs": dt, "worker": worker})
                log(worker, f"task{idx}: {scene} -> rate={rate} n={neps} ({dt}s)")
        except Exception as ex:
            log(worker, f"task{idx}: ERROR {ex}")
            record({"task_index": idx, "scenes": scenes, "status": "error", "error": str(ex)[:400], "worker": worker})
        finally:
            if server is not None:
                try:
                    server.terminate(); server.wait(timeout=20)
                except Exception:
                    server.kill()
                servers.pop(worker, None)
                time.sleep(3)
    log(worker, "WORKER DONE")

if __name__ == "__main__":
    log("main", f"plan={PLAN} workers={NW} trials={NTRIALS} already_done={len(done)}")
    threads = [threading.Thread(target=worker_fn, args=(w,), daemon=False) for w in range(NW)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    log("main", "ALL WORKERS DONE")
