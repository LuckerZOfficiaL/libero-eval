#!/usr/bin/env python3
"""EXP-83 step 7: deploy the trained hypernetwork on the 18 HELD-OUT receivers and evaluate by rollouts.

For each held-out receiver: embed its own demo frames with the frozen policy, get per-component
coefficients from the hypernetwork, merge the 68 donors, export the result as a normal adapter, and hand
it to the usual orchestrator for a 25-rollout commit. Compared against, on the same receivers:
  (a) the receiver alone                      -- its e44_base
  (b) random top-3 at the verified dose       -- the random-selection control
  (c) the behavioural probe's top-3           -- the method we actually deploy, the bar to beat
Exporting to a plain adapter matters: the rollout harness then treats this exactly like every other
merge, so nothing about the evaluation path differs from the numbers it is compared to.
"""
import argparse, json, pathlib, subprocess, sys
import numpy as np, torch
from safetensors import torch as st
REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "sweep")); sys.path.insert(0, str(REPO / "scripts"))
import unified_pool_driver as U
from exp83_merge import DonorBank
from exp83_hypernet import CoefHypernet, load_tokens

def export_adapter(name, recv_key, bank, w, order, rank=32, dev="cpu"):
    """Fold the merged delta into a NORMAL-sized adapter.

    The merged donor delta has effective rank K*16 = 1040. Writing it out verbatim produced a 12.5GB
    adapter (20x a normal one) and blew the disk quota on the first two receivers. It does not need that
    rank: truncate each stem's merged delta to `rank` with a randomized SVD, which is what the fleet's
    own adapters use anyway. The receiver's own residual is kept exact at 1.0, as the operator requires.
    """
    tf = U.fold(recv_key); tensors = {}
    for i, s_ in enumerate(order):
        Usig = U.USIG[s_]; Un = Usig / torch.linalg.norm(Usig, dim=0).clamp(min=1e-12); V = U.VS[s_]
        Bb, Ab = [Usig], [V.T]
        if s_ in tf:                                     # receiver keeps its own residual, exactly
            b, a = tf[s_]; M = Un.T @ b; N = a @ V
            Bb += [b, -(Un @ M)]; Ab += [a, N @ V.T]
        wi = w[i].to(dev)
        Bd = bank.B[s_].to(dev).float(); Ad = bank.A[s_].to(dev).float()
        D = (Bd * bank.expand(wi)[None, :]) @ Ad         # merged donor RESIDUAL (bank is residual-form)
        q = min(rank, min(D.shape) - 1)
        U_, S_, V_ = torch.svd_lowrank(D, q=q, niter=2)
        Bb.append((U_ * S_).cpu()); Ab.append(V_.T.cpu())
        tensors[f"{s_}.lora_A.weight"] = torch.cat(Ab, 0).contiguous()
        tensors[f"{s_}.lora_B.weight"] = torch.cat(Bb, 1).contiguous()
    p = REPO / f"results/analysis/decomp_adapters/{name}.safetensors"
    U._ensure_space(); st.save_file(tensors, str(p)); return str(p)


def main(a):
    free = [(torch.cuda.mem_get_info(i)[0], i) for i in range(torch.cuda.device_count())]
    dev = a.device or f"cuda:{max(free)[1]}"
    split = json.load(open(REPO / "results/analysis/exp83_split.json"))
    ck = torch.load(REPO / "results/analysis/exp83_hypernet.pt", map_location="cpu")
    donors, order = ck["donors"], ck["order"]
    print(f"[eval] checkpoint step {ck['step']} | {len(donors)} donors | {len(split['receivers'])} receivers", flush=True)
    bank = DonorBank(donors, device=dev, dtype=torch.float32)
    tok, band = load_tokens(order, donors, device=dev)
    Q = torch.tensor(np.load(REPO / "results/analysis/exp83_queries.npz")["Q"], dtype=torch.float32, device=dev)
    qidx = {k: i for i, k in enumerate(json.load(open(REPO / "results/analysis/exp83_queries.json")))}
    net = CoefHypernet(d_q=(a.d_q or Q.shape[1]), n_stems=len(order), K=bank.K, r=bank.r, max_scale=a.max_scale).to(dev)
    # energy matching needs each donor's residual norm and the operator's own ref
    net.rnorm.copy_(torch.tensor([U.resid_norm(k) for k in donors], dtype=torch.float32, device=dev))
    net.ref = 0.5 * float(np.mean([U.resid_norm(t) for t in range(0, 76, 8)]))
    print(f"[dose] ref {net.ref:.3f}, donor residual norms {float(net.rnorm.min()):.2f}-{float(net.rnorm.max()):.2f}", flush=True)
    net.load_state_dict(ck["state"]); net.eval()
    stem_ix = torch.arange(len(order), device=dev)
    todo = []
    for r in split["receivers"]:
        job = f"e91_{r['tag']}"
        if U.get(job, r["task"]) is not None: continue
        with torch.no_grad():
            w = net(Q[qidx[r["tag"]]], tok, stem_ix, band)
        key = r["key"] if isinstance(r["key"], int) else r["key"]
        todo.append((job, r, w.cpu()))
    print(f"[eval] {len(todo)} receivers to commit", flush=True)
    # External receivers live in a different SUITE with their own norm stats. Handing every receiver the
    # fleet's META (as the first version did) made all 6 goal-suite receivers parse_fail in seconds --
    # the same defect that cost EXP-81 its first 40 cells. Route each receiver through U.run with ITS
    # suite/norm, exactly as the union driver does.
    meta = {tag: (suite, norm) for tag, tk, task, suite, norm in U.targets()}
    CH = 2
    for i in range(0, len(todo), CH):
        built = []
        for job, r, w in todo[i:i + CH]:
            p = export_adapter(f"e83_{r['tag']}", r["key"], bank, w, order, dev=dev); built.append(p)
            suite, norm = meta[r["tag"]]
            U.run([{"name": job, "method": "average", "params": {}, "adapters": [],
                    "adapter_paths": [p], "eval_scenes": [r["task"]]}], U.COMMIT_R, suite, norm)
        for p in built: pathlib.Path(p).unlink(missing_ok=True)
    print("[eval] done", flush=True)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--d_q", type=int, default=0)      # 0 = take the dim from the query file
    p.add_argument("--max_scale", type=float, default=4.0); 
    p.add_argument("--device", default=None)
    main(p.parse_args())
