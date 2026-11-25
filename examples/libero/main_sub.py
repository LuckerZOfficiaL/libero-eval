"""Instrumented copy of main.py for ROUTES 2 and 3 (dense probe score / partial rollouts).

A copy, not an edit, so the live evaluation path stays untouched while EXP-32 is running. Control flow
is identical; two extra measurements per episode, both free (no additional simulator steps):

  PROGRESS (route 2): the binary success label is a Bernoulli draw whose 10-episode SD is ~14pp. That
    noise caps probe test-retest at rho=0.41 and therefore any learned selector at sqrt(0.41)=0.64, and
    it is why the probe's best donor repeats on only 1 target in 10. A dense score should carry the same
    information with far less variance. The task-agnostic one available to us: how far along the task's
    own DEMONSTRATION trajectory the policy got. Each step, find the nearest waypoint of the demo
    end-effector path and keep the furthest index reached, normalised to [0,1]. No per-task engineering,
    and the demos are already on the NAS.

  SUCCESS_STEP (route 3): successful episodes break early while failures always run to max_steps, so
    failures dominate the cost of a probe. Logging the step at which success occurs shows where an
    episode could be truncated without losing real successes.

Printed as PROGRESS:/SUCCESS_STEP: lines, one per episode, for the analysis to parse.
"""
import collections
import json
import dataclasses
import logging
import math
import pathlib

import imageio
import libero.libero as libero_pkg
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero import set_libero_default_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
from task_suite_task_map import LIBERO_TASK_MAP
import torch
import tqdm
import tyro

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data
_ORIGINAL_TORCH_LOAD = torch.load


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = (
        "libero_spatial"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    )
    task_name: str | None = None  # If set, evaluate only this single LIBERO task (overrides the repo task map)
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize i n sim
    num_trials_per_task: int = 50  # Number of rollouts per task
    save_videos: str = "all"  # videos per task: "all" | 0/"none" | k>0 (first k) | k<0 (last |k|, rolling)

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "data/libero/videos"  # Path to save videos

    seed: int = 7  # Random Seed (for reproducibility)
    state_offset: int = 0  # start from initial_states[offset]; lets a run use states never used for
    #                        selection or scoring, which is how we check the gains are not state-specific
    state_pool: int = 0  # EXP-106: if > 0, episode i uses initial_states[state_offset + (i % state_pool)] (never wraps
    #                      into states outside [offset, offset + pool), e.g. collection on 25..49, scoring on 0..24)
    dump_obs_path: str | None = None  # if set: per episode npz of camera frames + state (+ executed actions)
    dump_stride: int = 0  # 0 = frames at every replan only; k > 0 = frames every k steps AND every executed action
    ood_shift: float = 0.0  # EXP-81: metres to translate EVERY free-joint object's xy from its canonical
    #                         init pose, along one episode-seeded random direction (layout preserved).
    #                         The canonical 50 states span ~+-2.4 cm, so 0.05 / 0.10 is outside the demos.


def eval_libero(args: Args) -> None:
    # Video-saving policy (args.save_videos), per task/scene:
    #   "all"      -> keep every rollout (one file per episode)
    #   0 / "none" -> save nothing (also skips per-step frame capture, the main cost)
    #   k  (k > 0) -> keep the FIRST k rollouts (episodes 0..k-1); once k are saved, later ones don't record
    #   -k (k < 0) -> keep the LAST |k| rollouts, in rolling slots (episode_idx % |k|), overwriting older ones
    sv = str(args.save_videos).strip().lower()
    if sv == "all":
        vid_mode, vid_k = "all", 0
    elif sv in ("0", "none", "off", ""):
        vid_mode, vid_k = "none", 0
    else:
        _n = int(sv)
        vid_mode, vid_k = ("first" if _n > 0 else "last"), abs(_n)

    # Set random seed
    np.random.seed(args.seed)
    # Make LIBERO resolve benchmark assets from the installed package instead of a stale user config.
    set_libero_default_path(str(pathlib.Path(libero_pkg.__file__).resolve().parent))
    # LIBERO init-state files are trusted local pickle payloads and need the pre-PyTorch-2.6 load behavior.
    torch.load = lambda *load_args, **load_kwargs: _ORIGINAL_TORCH_LOAD(  # type: ignore[assignment]
        *load_args,
        **({"weights_only": False} | load_kwargs),
    )

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    _apply_repo_task_selection(task_suite, args.task_suite_name, args.task_name)
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    if args.task_suite_name == "libero_spatial":
        max_steps = 220  # longest training demo has 193 steps
    elif args.task_suite_name == "libero_object":
        max_steps = 280  # longest training demo has 254 steps
    elif args.task_suite_name == "libero_goal":
        max_steps = 300  # longest training demo has 270 steps
    elif args.task_suite_name == "libero_10":
        max_steps = 520  # longest training demo has 505 steps
    elif args.task_suite_name == "libero_90":
        max_steps = 400  # longest training demo has 373 steps
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    # Start evaluation
    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        # Start episodes
        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            logging.info(f"\nTask: {task_description}")

            # Reset environment
            env.reset()
            action_plan = collections.deque()

            # Set initial states
            _si = args.state_offset + (episode_idx % args.state_pool if args.state_pool > 0 else episode_idx)
            _st = np.array(initial_states[_si % len(initial_states)], copy=True)
            if args.ood_shift > 0:
                _sim = env.env.sim
                _rng = np.random.default_rng(81_000 + episode_idx)
                _ang = _rng.uniform(0, 2 * np.pi); _d = args.ood_shift * np.array([np.cos(_ang), np.sin(_ang)])
                for _j in range(_sim.model.njnt):
                    if _sim.model.jnt_type[_j] == 0:            # free joint => an object; state = [t] + qpos + qvel
                        _a = 1 + int(_sim.model.jnt_qposadr[_j]); _st[_a:_a + 2] += _d
                logging.info(f"OOD_SHIFT: {args.ood_shift:.3f} m, dir {np.degrees(_ang):.0f} deg")
            obs = env.set_init_state(_st)

            # Setup
            t = 0
            replay_images = []
            dump = {"img": [], "wrist": [], "state": [], "step": [], "act": [], "act_step": []}
            demo_xyz = _demo_traj(task)
            progress = 0.0
            success_step = -1
            # EXP-157: goal predicates of the BDDL conjunction, checked after every step (sub-goals of a composite task)
            _goals = env.env.parsed_problem["goal_state"]; _ever = [False] * len(_goals); max_sat = 0; _init = None
            # Record this episode's video? "first k" saves only episodes 0..k-1; "last k" always
            # records (rolling); "all" always; "none" never. Skips frame capture when not recording.
            record_video = (vid_mode == "all") or (vid_mode == "last") or (vid_mode == "first" and episode_idx < vid_k)

            logging.info(f"Starting episode {task_episodes+1}...")
            while t < max_steps + args.num_steps_wait:
                try:
                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    # and we need to wait for them to fall
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    # Get preprocessed image
                    # IMPORTANT: rotate 180 degrees to match train preprocessing
                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                    )
                    wrist_img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                    )

                    # Save preprocessed image for replay video (skip entirely when not recording)
                    if record_video:
                        replay_images.append(img)

                    if not action_plan:
                        # Finished executing previous action chunk -- compute new chunk
                        # Prepare observations dict
                        element = {
                            "observation/image": img,
                            "observation/wrist_image": wrist_img,
                            "observation/state": np.concatenate(
                                (
                                    obs["robot0_eef_pos"],
                                    _quat2axisangle(obs["robot0_eef_quat"]),
                                    obs["robot0_gripper_qpos"],
                                )
                            ),
                            "prompt": str(task_description),
                        }

                        if args.dump_obs_path and args.dump_stride == 0:
                            dump["img"].append(img); dump["wrist"].append(wrist_img)
                            dump["state"].append(element["observation/state"]); dump["step"].append(t)
                        # Query model to get action
                        action_chunk = client.infer(element)["actions"]
                        assert (
                            len(action_chunk) >= args.replan_steps
                        ), f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                        action_plan.extend(action_chunk[: args.replan_steps])

                    action = action_plan.popleft()
                    if args.dump_obs_path and args.dump_stride > 0:
                        if (t - args.num_steps_wait) % args.dump_stride == 0:
                            dump["img"].append(img); dump["wrist"].append(wrist_img); dump["step"].append(t)
                            dump["state"].append(np.concatenate((obs["robot0_eef_pos"], _quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])))
                        dump["act"].append(np.asarray(action, dtype=np.float32)); dump["act_step"].append(t)

                    # Execute action in environment
                    obs, reward, done, info = env.step(action.tolist())
                    _sat = [bool(env.env._eval_predicate(g)) for g in _goals]
                    _ever = [a or b for a, b in zip(_ever, _sat)]; max_sat = max(max_sat, sum(_sat))
                    _init = _sat if _init is None else _init   # after the first policy action ~ the initial state
                    if demo_xyz is not None:
                        _d = np.linalg.norm(demo_xyz - np.asarray(obs["robot0_eef_pos"]), axis=1)
                        progress = max(progress, (int(np.argmin(_d)) + 1) / len(demo_xyz))
                    if done:
                        success_step = t
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception as e:
                    logging.error(f"Caught exception: {e}")
                    break

            task_episodes += 1
            total_episodes += 1
            if args.dump_obs_path and dump["img"]:
                _dd = pathlib.Path(args.dump_obs_path); _dd.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(_dd / f"ep{episode_idx:03d}_s{_si:02d}_{'success' if done else 'failure'}.npz",
                                    **{k: np.asarray(v) for k, v in dump.items()}, done=np.array(bool(done)), state_idx=np.array(_si))

            # Save this episode's replay video per the save_videos policy (see top of eval_libero).
            # slot: rolling (episode_idx % |k|) for "last k" so only the last |k| survive; otherwise
            # unique per episode ("all" and "first k" write each once and never overwrite).
            if record_video:
                suffix = "success" if done else "failure"
                task_segment = task_description.replace(" ", "_")
                slot = (episode_idx % vid_k) if vid_mode == "last" else episode_idx
                video_dir = pathlib.Path(args.video_out_path)
                # Drop any existing video in this (task, slot) regardless of its success/failure
                # suffix, so each slot holds exactly one file and the per-task cap is respected.
                for stale in video_dir.glob(f"rollout_{task_segment}_{slot:03d}_*.mp4"):
                    stale.unlink()
                imageio.mimwrite(
                    video_dir / f"rollout_{task_segment}_{slot:03d}_{suffix}.mp4",
                    [np.asarray(x) for x in replay_images],
                    fps=10,
                )

            # Log current results
            logging.info(f"Success: {done}")
            logging.info(f"PROGRESS: {progress:.4f}")
            logging.info(f"SUCCESS_STEP: {success_step}")
            logging.info(f"SUBGOALS: {max_sat}/{len(_goals)}")
            _sg = pathlib.Path(args.video_out_path); _sg.mkdir(parents=True, exist_ok=True)
            with open(_sg / "subgoals.jsonl", "a") as _f:   # one line per episode; a retried scene appends again (last 10 win)
                _f.write(json.dumps({"ep": episode_idx, "si": int(_si), "done": bool(done), "max_sat": max_sat, "n_goal": len(_goals),
                                     "ever": _ever, "init": _init, "goals": [list(map(str, g)) for g in _goals]}) + "\n")
            logging.info(f"# episodes completed so far: {total_episodes}")
            logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

        # Log final results
        logging.info(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")

    logging.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")
    logging.info(f"Total episodes: {total_episodes}")


def _apply_repo_task_selection(task_suite, task_suite_name: str, task_name: str | None = None) -> None:
    if task_name is not None:
        # Single-task override: evaluate exactly one LIBERO task, bypassing the repo map.
        configured_tasks = [task_name]
    else:
        configured_tasks = LIBERO_TASK_MAP.get(task_suite_name)
        if configured_tasks is None:
            raise ValueError(f"Missing repo task mapping for suite: {task_suite_name}")

    available_tasks = {task.name: task for task in task_suite.tasks}
    missing_tasks = [task_name for task_name in configured_tasks if task_name not in available_tasks]
    if missing_tasks:
        raise ValueError(
            f"Configured tasks not found in installed LIBERO suite {task_suite_name}: {missing_tasks}"
        )

    task_suite.tasks = [available_tasks[task_name] for task_name in configured_tasks]
    task_suite.n_tasks = len(task_suite.tasks)
    logging.info("Using %d repo-selected tasks for %s", task_suite.n_tasks, task_suite_name)


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


_DEMO_CACHE = {}


def _demo_traj(task):
    """End-effector xyz path from this task's demonstrations: the reference curve for progress.

    Keyed by the bddl file stem, which is how the demo hdf5 files are named. Returns None when the demo
    is missing, in which case progress stays 0 and only SUCCESS_STEP is meaningful for that task.
    """
    import pathlib as _pl
    stem = _pl.Path(str(getattr(task, "bddl_file", "") or "")).stem
    if not stem:
        return None
    if stem in _DEMO_CACHE:
        return _DEMO_CACHE[stem]
    hits = list(_pl.Path("/media/pinas/palma/inspire_4suite_regenerated_hdf5").glob(f"*/{stem}_demo.hdf5"))
    traj = None
    if hits:
        try:
            import h5py
            with h5py.File(hits[0], "r") as h:
                d = h["data"] if "data" in h else h
                traj = np.asarray(d[sorted(d.keys())[0]]["obs/ee_pos"], dtype=np.float32)
        except Exception as e:
            logging.warning(f"demo trajectory unavailable for {stem}: {e}")
    _DEMO_CACHE[stem] = traj
    return traj


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_libero)
