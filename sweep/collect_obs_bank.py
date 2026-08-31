#!/usr/bin/env python
"""Collect a fixed observation bank for offline activation probing.

For every scene in the repo task map: init the LIBERO env exactly as the eval harness does,
settle with dummy actions, and record N observations (agentview+wrist images preprocessed
identically to eval, 8-dim state, prompt = task language). Saved to results/analysis/obs_bank.npz.

No policy involved — the bank is policy-independent, so every adapter is probed on identical inputs.
"""
import collections
import json
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "examples" / "libero"))
import main as libmain  # reuse _get_libero_env, _quat2axisangle, preprocessing constants

import libero.libero as libero_pkg
from libero.libero import benchmark, set_libero_default_path
from openpi_client import image_tools
import torch

OBS_PER_SCENE = 2
SETTLE = 10          # match args.num_steps_wait
EXTRA_GAP = 15       # steps between the two snapshots

_ORIG_LOAD = torch.load
torch.load = lambda *a, **k: _ORIG_LOAD(*a, **({"weights_only": False} | k))
set_libero_default_path(str(pathlib.Path(libero_pkg.__file__).resolve().parent))

plan = json.load(open(REPO / "sweep" / "plan.json"))["adapters"]
scene2lang = {}
for a in plan:
    for s in a["scenes"]:
        scene2lang.setdefault(s, a["language"])

suite = benchmark.get_benchmark_dict()["libero_90"]()
name2id = {suite.get_task(i).name: i for i in range(suite.n_tasks)}
# repo task map may rename scenes; match by name prefix
def find_task(scene):
    if scene in name2id:
        return name2id[scene]
    for n, i in name2id.items():
        if scene.startswith(n) or n.startswith(scene):
            return i
    raise KeyError(scene)

imgs, wrists, states, prompts, scene_names = [], [], [], [], []
scenes = sorted(scene2lang)
for si, scene in enumerate(scenes):
    tid = find_task(scene)
    task = suite.get_task(tid)
    init_states = suite.get_task_init_states(tid)
    env, desc = libmain._get_libero_env(task, libmain.LIBERO_ENV_RESOLUTION, 7)
    env.reset()
    obs = env.set_init_state(init_states[0])
    t = 0
    snaps = []
    while len(snaps) < OBS_PER_SCENE:
        obs, *_ = env.step(libmain.LIBERO_DUMMY_ACTION)
        t += 1
        if t == SETTLE or t == SETTLE + EXTRA_GAP:
            img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
            wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
            img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, 224, 224))
            wrist = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist, 224, 224))
            state = np.concatenate((obs["robot0_eef_pos"],
                                    libmain._quat2axisangle(obs["robot0_eef_quat"]),
                                    obs["robot0_gripper_qpos"]))
            snaps.append((img, wrist, state))
    env.close()
    for img, wrist, state in snaps:
        imgs.append(img); wrists.append(wrist); states.append(state.astype(np.float32))
        prompts.append(scene2lang[scene]); scene_names.append(scene)
    if (si + 1) % 10 == 0:
        print(f"{si+1}/{len(scenes)} scenes", flush=True)

out = REPO / "results" / "analysis" / "obs_bank.npz"
np.savez_compressed(out, img=np.stack(imgs), wrist=np.stack(wrists), state=np.stack(states),
                    prompt=np.array(prompts), scene=np.array(scene_names))
print(f"saved {out}: {len(imgs)} observations from {len(scenes)} scenes")
