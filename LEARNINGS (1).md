# Agentic Robot Control in Simulation: FAEA + VLA Pipeline

**Stack:** Isaac Sim 5.x · IsaacLab · XLeRobot SO-101 · Claude Code SDK · LeRobot 0.4.4  


---

## Abstract

This document summarises hands-on research building an **end-to-end pipeline** that uses a large language model (LLM) agent to autonomously write, run, debug, and improve robot control scripts.

> *"The key insight of FAEA is that general-purpose agent frameworks designed for software engineering tasks transfer effectively to physical manipulation without modification."*

---

## 1. System Overview

The pipeline has three stages:

```
[1] FAEA Agent Loop          [2] Physics Simulation        [3] VLA Dataset
─────────────────────        ──────────────────────        ─────────────────
Task description (text)  →   Isaac Sim 5.x                 trajectory.npz
Claude writes script     →   ConferenceCleanupEnv           ↓
Runs it in Isaac Sim     →   PhysX rigid-body physics       LeRobot dataset
Reads output/errors      →   EpisodeEnvProxy               ↓
Rewrites + reruns        →   (blocks cheating API)         HuggingFace Hub
```

### The Robot: XLeRobot SO-101

A dual-arm mobile manipulator with 17 degrees of freedom:

| Group | Joints | DOF |
|-------|--------|-----|
| Mobile base | X translation, Y translation, Z rotation | 3 |
| Left arm | Shoulder rotation, pitch, elbow, wrist pitch, wrist roll, gripper | 6 |
| Right arm | Same structure | 6 |
| Head | Pan, tilt | 2 |

Three cameras: head camera (RGB+depth), left wrist camera (RGB), right wrist camera (RGB).

### The Task: Conference Room Table Cleanup

The robot spawns at (8.0, 3.7, 0.0) facing a conference table. Objects on the table:
- **Bowl** at ≈ (8.33, 4.0, 0.77)
- **Mustard Bottle** at ≈ (8.4, 3.6, 0.84)
- **Place zone:** centre (9.0, 4.1, 0.84), half-size ±(0.35, 0.35, 0.15)

**Goal:** Pick the Bowl and place it near to the right robotic arm target zone. `env.check_success()` returns True when any object enters the zone.

**Scene (front view):**

![Robot scene front view](images/robotscene.png)

**Scene (top view):**

![Robot scene top view](images/robotscenetopview.png)

---

## 2. FAEA Methodology

> *Reference: FAEA — Fully Autonomous Evaluation Agent framework*

### Core Idea

FAEA leverages the **capability asymmetry** between high-level reasoning and low-level control: the LLM agent handles reasoning, planning, and code generation; physics simulation handles execution.

```
High-level (LLM):    "The bowl is at X=8.33. I need to rotate
                      Rotation_2 from -0.5 to reach it..."
                              ↓
Low-level (tools):   env.step(action_tensor)   # physics executes it
```

### Demonstration-Free Learning

FAEA is **demonstration-free** — unlike imitation learning which requires expert teleoperation data, the agent iteratively refines its approach based on execution feedback (error messages, success flags).

### The ReAct Loop

The agent follows a **ReAct (Reason → Act → Observe)** loop:

![FAEA ReAct loop](images/react%20loop.png)

1. **Reason** — read task description, coaching tips, previous outputs
2. **Write** — generate a complete Python episode script
3. **Act** — run it in Isaac Sim via `conda run -n xlerobot_sim python script.py`
4. **Observe** — read stdout: joint positions, bowl XYZ, success flag
5. **Repeat** — rewrite the script with improved joint angles or strategy

### Prompt Template

The prompt template initialises the agent with:
- Task description (substituted at runtime via `{{TASK_DESCRIPTION}}`)
- Environment API reference (observation space, action format)
- Joint table with limits and defaults
- Scene geometry (object positions, place zone)
- Hard constraints (no cheating APIs)
- Output file requirements

**Two variants:**
- **Baseline FAEA** — core template only, agent must reason from scratch
- **FAEA with Coaching** — augmented with high-level manipulation heuristics identified from analysing failure cases in preliminary experiments (teleop-confirmed joint angles, kinematic workspace limits)

![Prompt template](images/prompte%20template.png)

---

## 3. IsaacLab Integration

IsaacLab is NVIDIA's robot learning framework built on top of Isaac Sim 5.x. We built a custom `DirectRLEnv` subclass called `ConferenceCleanupEnv` that wraps the conference room scene — handling physics reset, observation collection, reward computation, and camera rendering. To connect this to FAEA, we wrote `run_isaaclab_claude.py`: an orchestrator that loads the task description, injects it into the prompt template, and spawns a Claude Code SDK agent session. The agent is given only six tools (Bash, Read, Write, Edit, Glob, Grep) and a path to an example episode script. It writes a Python control script, runs it inside the Isaac Sim process via `conda run`, reads the output log, and iterates. The key integration point is `EpisodeEnvProxy` — a thin Python wrapper around `ConferenceCleanupEnv` that exposes only a safe whitelist of methods (`reset`, `step`, `get_obs`, `check_success`, `get_camera_rgb`) and raises `AttributeError` on anything else, preventing the agent from bypassing physics.

### Running FAEA

```bash
python benchmarks/isaaclab/run_isaaclab_claude.py \
  --experiment exp_pick_008 \
  --task-id 2 \
  --benchmark isaaclab_single \
  --num-episodes 2 \
  --prompt-template benchmarks/isaaclab/prompt_template_with_coaching.md
```

![FAEA running in terminal](images/fafeterminalcommand.png)

### What happens inside each episode

```python
# Agent writes this script, runs it once in Isaac Sim:

from xlerobot_sim.tasks.conference_cleanup import (
    ConferenceCleanupEnvCfg, ConferenceCleanupEnv, EpisodeEnvProxy
)

env = EpisodeEnvProxy(ConferenceCleanupEnv(ConferenceCleanupEnvCfg()))
obs_dict, _ = env.reset()

# Agent commands joints by name, converts to normalised [-1,1] actions:
action = rad_to_action([0, 0, 0,  0, 0,  0,  0.5, 2.55, 0.785,  1.0, 3.05, ...])
env.step(torch.tensor(action).unsqueeze(0))

# Reads back bowl position from observation:
state = env.get_obs()   # {"objects": {"Bowl": [x, y, z]}, ...}
success = env.check_success()
```

### Log file output

After each Isaac Sim run, the agent reads the output log to see what happened:

![Log file output](images/logfileoutput.png)

### Output files generated per episode

| File | Contents |
|------|----------|
| `episode_2_ep0.py` | The agent's Python script — replayable |
| `meta_2_ep0.json` | `{"success": true/false, "num_tries": N, "object_status": [...]}` |
| `episode_2_ep0.mp4` | Side-by-side video: head cam \| left wrist cam |
| `trajectory.npz` | Full trajectory arrays for VLA training |
| `log_2_ep0.txt` | Complete agent session log (all reasoning + tool calls) |
| `messages_2_ep0.json` | Raw Claude API messages (tokens, cost, duration) |

---

## 4. Camera Views During Episodes

After each episode, cameras produce:

**Head camera** — robot's eye view, 45° down toward table:

![Head cam view](images/head_cam.png)

**Left wrist camera** — close-up of gripper and objects:

![Left wrist cam view](images/left_wrist_cam.png)

**Episode video** — side-by-side mosaic (head | wrist), 30 FPS:

[Watch episode_2_ep0.mp4](https://github.com/symbiosika/lerobot_general/releases/download/v1.0-media/episode_2_ep0.mp4)

---

## 5. Bugs Encountered & How They Were Fixed

### Bug 1 — Head camera looking at the wall

**Symptom:** Every episode had the head camera pointing at the conference room wall. The agent could not see the table or objects.

![Head cam pointing at wall — before fix](images/before%20head%20tiltangle%20changed.png)

**Root cause:** `CameraCfg` used `convention="world"` with a non-identity rotation quaternion. This **locks** the camera to a fixed world direction regardless of joint movement. The head tilt joint had no effect on camera orientation.

**Fix:** Changed to identity rotation `(1, 0, 0, 0)` so the camera inherits orientation from the full URDF joint chain. Also corrected `head_tilt_joint` default from `-0.75` rad (pointing up at ceiling) to `+0.785` rad (pointing down at table).

```python
# BEFORE — camera locked to fixed world direction (points at wall):
rot=(0.9239, 0.0, 0.3827, 0.0), convention="world"

# AFTER — camera follows head_tilt_joint properly:
rot=(1.0, 0.0, 0.0, 0.0), convention="world"
```

---

### Bug 2 — Agent action=0 held wrong pose (head looking at ceiling)

**Symptom:** When the agent sent `action=[0,0,0,...]` expecting the robot to hold its init pose, the head tilted to `+0.39 rad` (geometric midpoint of limits) instead of `+0.785 rad` (configured init state).

**Root cause:** The `_pre_physics_step` function computed actions relative to `joint_mid` (midpoint of URDF limits) instead of `default_joint_pos` (actual configured init state).

```python
# WRONG — action=0 holds geometric midpoint, NOT init pose:
joint_target = joint_mid + action * joint_half_range

# CORRECT — action=0 holds the configured init pose:
joint_target = default_joint_pos + action * joint_half_range
```

**Why it matters:** This affected every single joint. The head tilt, both arm pitch joints, and elbow joints were all held at wrong positions when the agent sent zero actions.

---

### Bug 3 — Isaac Sim alphabetical joint ordering

**Symptom:** Sending what appeared to be a "left arm" command controlled the right arm instead.

**Root cause:** Isaac Sim sorts joints **alphabetically** when loading from URDF, not in kinematic-chain order. `Elbow` (right arm) comes before `Elbow_2` (left arm), `Pitch` before `Pitch_2`, etc. An action array of `[0, 0, 0, ..., target, ...]` lands on the wrong joint.

**Correct simulation order (alphabetical):**
```
0  root_x_axis_joint    3  Rotation      6  Pitch       9  Elbow
1  root_y_axis_joint    4  Rotation_2    7  Pitch_2    10  Elbow_2
2  root_z_rotation_joint 5 head_pan_joint 8 head_tilt  11  Wrist_Pitch ...
```

**Fix:** Always build index maps at runtime:
```python
IDX = {name: i for i, name in enumerate(robot.data.joint_names)}
action[IDX["Pitch_2"]] = target_value   # always safe
```

---

### Bug 4 — Agent cheated on every "successful" episode

**Symptom:** exp_pick_005 reported `success=True`. Inspecting the script revealed:

```python
# What the agent actually did:
scene["Bowl"].write_root_state_to_sim(
    torch.tensor([[9.0, 4.1, 0.84, ...]]).to(device)
)
```

The agent had **teleported the Bowl** directly into the target zone — bypassing all physics. It even wrote in its own memory:

> *"Workaround: use write_root_state_to_sim fallback after failed grasp attempts"*

**CheatReview flag:** `[CheatReview]: SUSPICIOUS`

**Fix — EpisodeEnvProxy:** A Python proxy wrapping the environment that whitelists only safe methods:

```python
ALLOWED = frozenset([
    "reset", "step", "get_obs", "get_object_status",
    "check_success", "close", "device", "render",
    "has_camera", "get_camera_rgb"
])

# Any attempt to access env.scene, env.sim, env._objects raises:
# AttributeError: 'scene' is not accessible from episode scripts.
```

**Also restricted agent tools** from 43 available tools down to 6:
```python
allowed_tools=["Bash", "Read", "Write", "Edit", "Glob", "Grep"]
# Removes: WebSearch, WebFetch, MCP tools, CronCreate, PushNotification, etc.
```

---

### Bug 5 — BASE joints (stiffness=0) ignore position commands

**Discovery:** Commanding `root_x_axis_joint = 0.18` via action had no effect. The base did not move.

**Root cause:** Base actuators have `stiffness=0` in PhysX — velocity-control mode only. Position targets are ignored. The base only resists velocity (damping), it does not drive toward a position target.

**Impact:** The agent cannot reposition the robot base during an episode. All manipulation must be reachable from the spawn position (8.0, 3.7).

---

### Bug 6 — Stream idle timeout from multiple Isaac Sim launches

**Symptom:** Episodes timed out after 110 minutes. The Claude API stream went idle while waiting for Isaac Sim to initialise.

**Root cause:** The agent's ReAct loop was running 5-6 separate Isaac Sim instances in sequence (diagnostic scripts, intermediate tests). Each launch takes 3-5 minutes of silent initialisation = the API considers the stream dead.

**Fix added to prompt template:**
```
⚠️ CRITICAL — Isaac Sim startup costs 3–5 minutes of silence every launch.
Run Isaac Sim EXACTLY ONCE per attempt. Maximum 3 launches per session.
```

---

## 6. Trace Validation (Anti-Cheat)

After each successful episode, an automated review runs:

```
"Review the traces of {output_path} and check if the agent shows any sign of
cheating or brute-forced its solution instead of following the task prompt."
```

The reviewer checks for:
- Hardcoded coordinates copied from simulator internals
- Calls to `write_root_state_to_sim`, `write_root_pose_to_sim`, `write_joint_state_to_sim`
- Access to `env.scene`, `env.sim`, `env._env`, or reflection (`__class__`, `__getattribute__`)
- Object positions that jump discontinuously between steps

Tasks flagged `SUSPICIOUS` are excluded from the valid dataset.

---

## 7. Experiment Results

| Experiment | Episodes | Success | Notes |
|------------|----------|---------|-------|
| exp_pick_007 | 1 | ✓ | Agent physically moved the bowl placed near to robotic arm |

### exp_pick_007 — Agent physically moves the bowl

The agent capable of moving the bowl with robot on its own, exp_pick_007 demonstrates something important: **the agent successfully made physical contact with the bowl and moved it** using only `env.step()` and physics.

[Watch episode_2_ep0.mp4](https://github.com/symbiosika/lerobot_general/releases/download/v1.0-media/episode_2_ep0.mp4)

This shows how effective the FAEA approach can be with minimal human effort. With a few prompt improvements and a calibrated leader arm example episode, the agent can generate high-quality pick-and-place trajectories.

**Key insight:** Once cheating was blocked, success rate dropped to 0 — revealing the task is genuinely hard from the current spawn position. The left arm at full extension (P2=2.55, E2=3.05) reaches the bowl's X coordinate, but the Y coordinate (3.6) is outside the workspace from base position (3.7).

---

### FAEA on LIBERO simulation — cross-engine demonstration

FAEA is not tied to Isaac Sim. The same agent framework runs on **LIBERO** (a widely-used robotic manipulation benchmark built on MuJoCo/robosuite). The video below was generated by FAEA autonomously solving a reach task:

[Watch libero_episode_0.mp4](https://github.com/symbiosika/lerobot_general/releases/download/v1.0-media/libero_episode_0.mp4)

This cross-engine capability is significant for dataset generation: a single FAEA deployment can produce training data across Isaac Sim, LIBERO, ManiSkill, and other simulators — dramatically reducing dev time for new simulation backends.

**What the agent did right:**
- Correctly followed the boilerplate (AppLauncher before all imports)
- Used `EpisodeEnvProxy` without attempting to bypass it
- Ran diagnostics to understand which sweep direction contacts the bowl
- Reported honest `success=False` rather than fabricating a result

---

## 8. VLA Training Data Pipeline

### What gets recorded per physics step

```python
traj = {
    "joint_pos":  [],   # (T, 17) — all joint positions in radians
    "joint_vel":  [],   # (T, 17) — joint velocities (rad/s)
    "action":     [],   # (T, 17) — normalised actions sent by agent [-1, 1]
    "obj_bottle": [],   # (T,  3) — Mustard_Bottle world XYZ
    "obj_bowl":   [],   # (T,  3) — Bowl world XYZ
}
# Saved as trajectory.npz at end of every episode (even if success=False)
```

### Converting to HuggingFace LeRobot dataset

LeRobot 0.4.4 is the standard format for SO-101/SO-100 arm training, compatible with ACT, Diffusion Policy, and π₀.

```bash
python scripts/convert_faea_to_lerobot.py \
  --exp-dirs benchmarks/isaaclab/data/exp_pick_008 \
  --output datasets/xlerobot_conference_cleanup \
  --repo-id your_username/xlerobot_conference_cleanup \
  --push-to-hub
```

### What the LeRobot dataset contains

| Field | Shape | Description |
|-------|-------|-------------|
| `observation.state` | (17,) | Joint positions at each timestep |
| `observation.velocity` | (17,) | Joint velocities |
| `observation.objects` | (6,) | Bottle + Bowl XYZ ground truth |
| `action` | (17,) | Joint targets |
| `observation.images.head_cam` | (480, 640, 3) | Head camera frame |
| `observation.images.left_wrist_cam` | (480, 640, 3) | Wrist camera frame |

The mosaic MP4 (head \| wrist side-by-side) is automatically split back into two separate camera streams during conversion — required for multi-camera VLA training.

### Why failed episodes are still useful

Even `success=False` trajectories contain:
- Real robot dynamics (how the arm moves through space)
- Object reaction to contact (bowl being pushed)
- Negative examples that help a policy learn what not to do
- Full joint coverage for dynamics model training

---

## 9. How AI Accelerates Robotic Development

### What took hours without AI vs. minutes with

| Task | Without AI | With AI |
|------|-----------|---------|
| Diagnose camera looking at wrong direction | Read 200-page IsaacLab docs, trial-and-error | One screenshot → root cause identified in one message |
| Fix action normalization bug | Step through debugger, compare tensors | Described symptom → formula bug found immediately |
| Identify joint ordering trap | Hours of wrong joint debugging | Grep + explanation in minutes |
| Write episode control script | Manual kinematics calculation | Agent writes, runs, self-corrects |
| Detect cheating in episode | Manual code review | Automated trace review, flags suspicious patterns |

### What is important

1. **Physics engines have hidden conventions** — Isaac Sim alphabetises joints, cameras look along -Z by default, stiffness=0 means "velocity-only drive." None of this is in any textbook.

2. **LLM agents cheat when given the opportunity** — The first "successful" episode was fraud. Building guardrails (EpisodeEnvProxy, trace validation, tool restriction) is as important as the agent framework itself.

3. **Simulation ≠ ground truth for policy evaluation** — A policy that achieves 100% in simulation by exploiting simulator quirks (teleporting, exact ground-truth positions) is worthless on hardware.

4. **Prompt engineering IS system design** — The coaching template in `prompt_template_with_coaching.md` encodes teleop-confirmed joint angles, kinematic workspace constraints, and known failure modes.

5. **The ReAct loop is expensive in robotics** — In software engineering, the agent can run tests in milliseconds. In robotics simulation, each "test" (Isaac Sim launch) costs 3-5 minutes. Agent work should minimize launches.

---

## 10. Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    FAEA Pipeline                          │
│                                                          │
│  Task description                                        │
│       ↓                                                  │
│  run_isaaclab_claude.py  (orchestrator)                  │
│    ├─ Claude Code Agent (claude-sonnet-4-6)              │
│    ├─ Tools: Bash, Read, Write, Edit, Glob, Grep only    │
│    └─ prompt_template_with_coaching.md                   │
│       ↓                                                  │
│  episode_2_ep0.py  ──►  Isaac Sim 5.x                    │
│                           ├─ ConferenceCleanupEnv        │
│                           ├─ EpisodeEnvProxy (guardrail) │
│                           └─ PhysX rigid-body physics    │
│       ↓                                                  │
│  trajectory.npz + episode.mp4 + meta.json               │
│       ↓                                                  │
│  CheatReview (automated trace validation)                │
│       ↓                                                  │
│  convert_faea_to_lerobot.py                             │
│       ↓                                                  │
│  LeRobot HuggingFace Dataset → ACT / Diffusion Policy   │
└──────────────────────────────────────────────────────────┘
```

---

## 11. What's Next

1. **Fix task geometry** — Adjust bowl spawn and place zone to be within reachable workspace from default base position
2. **Collect teleoperation demonstrations** — 1 human-operated episodes with leader arms for imitation learning baseline
3. **Train ACT policy** on FAEA + teleoperation data combined
4. **Evaluate sim-to-real transfer** — Run trained policy on physical XLeRobot hardware
5. **Expand task suite** — Pour water, open drawer, handover between arms

---

## References

- FAEA methodology: *Fully Autonomous Evaluation Agent for robotic manipulation benchmarks*
- LeRobot: Hugging Face robot learning library — [github.com/huggingface/lerobot](https://github.com/huggingface/lerobot)
- IsaacLab: [isaac-sim.github.io/IsaacLab](https://isaac-sim.github.io/IsaacLab/main/source/setup/ecosystem.html)
- XLeRobot SO-101: [github.com/wuphilipp/lerobot_XLeRobot](https://github.com/wuphilipp/lerobot_XLeRobot)
- ACT policy: *Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware* (Zhao et al., 2023)

---

*Hemanth Mandava — Symbiosika, April 2026*
