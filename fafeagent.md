# Complete Reading Material — FAEA + IsaacLab + Robot Control
# Everything you need to understand the project deeply

---

## Part 1: What is Claude doing here?

### 1.1 Claude as a Code-Writing Agent

Claude is not controlling the robot directly. It is acting as a **programmer** that writes Python scripts. Think of it like this:

```
You give Claude a task in English:
  "Pick the Bowl from the table and place it in the target zone"

Claude writes Python code:
  smooth_move(DEFAULT_RAD, at_bowl_pose, steps=120)
  close_jaw()
  lift_arm()
  swing_to_place_zone()
  open_jaw()

Claude runs that code in Isaac Sim.
Claude reads what happened (bowl moved? success? error?).
Claude rewrites the code and tries again.
```

Claude never touches the robot directly. It only writes and runs Python scripts. The robot moves because of physics, not because Claude is computing trajectories.

---

### 1.2 What is a ReAct Loop?

ReAct stands for **Reason → Act → Observe**. It is the core pattern Claude follows.

```
┌─────────────────────────────────────────────────┐
│                  REACT LOOP                      │
│                                                  │
│  1. REASON                                       │
│     "Bowl is at X=8.33. Place zone at X=9.0.    │
│      I need to move bowl +0.67m in X direction.  │
│      Last attempt pushed it the wrong way."      │
│                 ↓                                │
│  2. ACT                                          │
│     Write episode script.                        │
│     Run it: conda run python episode.py          │
│                 ↓                                │
│  3. OBSERVE                                      │
│     Read stdout:                                 │
│     "Bowl after push: (8.26, 3.65, 0.77)"       │
│     "Success: False"                             │
│                 ↓                                │
│  4. REASON AGAIN                                 │
│     "Bowl went -X not +X. Sweep direction wrong. │
│      Try R2 from +1.5 → -0.5 instead."          │
│                 ↓                                │
│     (loop repeats up to 3 Isaac Sim launches)    │
└─────────────────────────────────────────────────┘
```

**Why is this powerful?**
Claude accumulates knowledge across turns. Each run tells it something new about the physics. It is learning in context — not by updating neural network weights, but by reasoning about what it observed and changing its code.

---

### 1.3 What is High-Level Planning?

High-level planning means thinking in **tasks and goals**, not in motor commands.

```
HIGH LEVEL (what Claude thinks):             LOW LEVEL (what actually happens):

"Reach the bowl"                    →        Pitch_2 = 2.55 rad
"Grip it"                           →        Jaw_2 close from 1.70 → 0.20
"Lift high so it clears the table"  →        Pitch_2 = 1.50, Elbow_2 = 1.80
"Swing toward the target"           →        Rotation_2 = -0.80
"Lower and release"                 →        Pitch_2 = 2.20, Jaw_2 = 1.70
```

Claude does the high-level planning. The `rad_to_action()` function does the conversion to motor commands. The physics engine executes the actual movement.

**This separation is the key idea:**
- Claude does not need to know how PhysX solves rigid body dynamics
- Isaac Sim does not need to understand what "pick up the bowl" means
- They communicate through a shared interface: the action vector

---

### 1.4 What tools does Claude have?

In this project Claude is restricted to 6 tools only:

| Tool | What it does | Example use |
|------|-------------|-------------|
| `Bash` | Run shell commands | Launch Isaac Sim, read logs |
| `Read` | Read any file | Read example scripts, read error output |
| `Write` | Create new files | Write the episode Python script |
| `Edit` | Edit existing files | Fix bugs in the script |
| `Glob` | Find files by pattern | Find all .py files |
| `Grep` | Search inside files | Find where a function is defined |

**Why restrict to 6?** Originally Claude had 43 tools including web search, email, calendar etc. It was using web search instead of solving the task. Restricting to 6 keeps Claude focused.

---

## Part 2: Isaac Sim — The Physics Environment

### 2.1 What is Isaac Sim?

Isaac Sim is NVIDIA's **physics simulator for robotics**. It uses:
- **USD (Universal Scene Description)** — the file format for 3D scenes (same as Pixar uses for movies)
- **PhysX** — NVIDIA's physics engine (rigid bodies, contact forces, gravity, friction)
- **RTX renderer** — photorealistic rendering for camera images

Think of it as a very accurate video game engine where the physics matches real life closely enough that a robot policy trained in simulation can transfer to a real robot.

### 2.2 What is IsaacLab?

IsaacLab is a **framework on top of Isaac Sim** for robot learning. It provides:
- Standard environment APIs (like OpenAI Gym)
- Easy robot loading from URDF/USD files
- Observation and action management
- Reward computation
- Episode reset

Our environment `ConferenceCleanupEnv` is an IsaacLab `DirectRLEnv` — a class we wrote that inherits from IsaacLab and defines our specific task.

### 2.3 The Scene — What's in the Simulation

```
World coordinate system:
  X = left/right (East/West)
  Y = forward/backward (North/South)
  Z = up/down

Conference room scene:
  Table surface at Z = 0.74 m
  
  XLeRobot spawn:  (8.0,  3.7, 0.0)  facing +Y
  Bowl:            (8.33, 4.0, 0.77)
  Mustard Bottle:  (8.4,  3.6, 0.84)
  Place zone:      center (9.0, 4.1, 0.84), half-size ±(0.35, 0.35, 0.15)
```

**Visual layout (top-down view):**

```
         +Y (forward)
          ↑
          │    [Table surface]
          │    ○ Bowl (8.33, 4.0)
          │    ○ Bottle (8.4, 3.6)
          │                        □ Place zone (9.0, 4.1)
[Robot]───┼─────────────────────────────────────── +X (right)
(8.0,3.7) │
```

---

### 2.4 How Does the Environment Know Where Things Are?

Every physics object in Isaac Sim has a **rigid body** with a position and orientation tracked by PhysX every step.

```python
# Inside ConferenceCleanupEnv, each step:
bowl_pos   = self.scene["Bowl"].data.root_pos_w      # (x, y, z) in world frame
bottle_pos = self.scene["Mustard_Bottle"].data.root_pos_w

# These get packed into the observation vector:
obs = torch.cat([
    robot.data.joint_pos,   # 17 values — all joint angles
    robot.data.joint_vel,   # 17 values — all joint velocities
    bottle_pos,             # 3 values  — XYZ in world
    bowl_pos,               # 3 values  — XYZ in world
], dim=-1)
# Total: 40 values per step
```

**Claude reads these coordinates** by printing them in the episode script:
```python
state = env.get_obs()
bowl = state["objects"]["Bowl"]   # → [8.33, 4.0, 0.77]
print(f"Bowl: {bowl}")
```

The agent does not "see" the bowl through the camera to get coordinates — it reads them directly from the physics engine. Camera images are recorded for the video and VLA training data but the agent uses the numerical coordinates for decision-making.

---

### 2.5 The Observation Space (40 numbers per step)

Every time `env.step(action)` is called, it returns an observation:

```
Index    Content                    Units
──────   ──────────────────────── ────────
0–16     joint positions (17)       radians
17–33    joint velocities (17)      rad/s
34–36    Mustard_Bottle XYZ (3)     meters
37–39    Bowl XYZ (3)               meters
──────────────────────────────────────────
Total:   40 values
```

This is everything the agent needs to know about the current state of the world.

---

## Part 3: From High-Level Plan to Motor Commands

### 3.1 The Robot's Joints

The XLeRobot has **17 joints**. Each joint is a rotational axis with physical limits:

```
Joint name         What it moves              Limits (rad)   Default
─────────────────  ─────────────────────────  ─────────────  ────────
root_x_axis_joint  base moves left/right      -10 to +10     0.0
root_y_axis_joint  base moves forward/back    -10 to +10     0.0
root_z_rotation    base rotates (yaw)         -π to +π       0.0
Rotation_2         left shoulder side-to-side -2.1 to +2.1   0.0
Pitch_2            left shoulder up/down      -0.1 to +3.45  0.775
Elbow_2            left elbow bend            -0.2 to +3.14  1.379
Wrist_Pitch_2      left wrist tilt            -1.8 to +1.8   0.0
Wrist_Roll_2       left wrist spin            -π to +π       0.0
Jaw_2              left gripper               0.0 to 1.70    0.0
head_tilt_joint    head up/down               -0.79 to +1.57 0.785
... (right arm mirrors left arm)
```

**Jaw values:** 0.0 = fully closed, 1.70 = fully open.

---

### 3.2 The Action Space — How Claude Commands Joints

Claude cannot command joints in radians directly — it sends a **normalised action vector** where every value is between -1.0 and +1.0.

**Why normalise?** Reinforcement learning algorithms and neural networks work best with inputs in a consistent range. -1 to +1 is the standard.

**The conversion formula:**
```
action = (target_radians - default_radians) / half_range

where:
  default_radians = the joint's default (init) position
  half_range      = (upper_limit - lower_limit) / 2
```

**Example — commanding Pitch_2 to reach the bowl:**
```
Target: Pitch_2 = 2.55 rad  (teleop-confirmed bowl contact)
Default: Pitch_2 = 0.775 rad
Half range: (3.45 - (-0.1)) / 2 = 1.775

action = (2.55 - 0.775) / 1.775 = 1.775 / 1.775 = +1.0
```

So `action[7] = +1.0` commands the left shoulder to full forward extension. This is why we say "action=+1.0 reaches the bowl."

**The Python function:**
```python
DEFAULT_RAD = [0,0,0, 0,0, 0, 0.5,0.775, 0.785, 1.0,1.379, 0,0, 0,0, 0,0]
JOINT_LIMITS = [(-10,10),(-10,10),(-3.14,3.14), ...]

def rad_to_action(rad_values):
    actions = np.zeros(17)
    for i, (lo, hi) in enumerate(JOINT_LIMITS):
        half = (hi - lo) / 2.0
        actions[i] = np.clip((rad_values[i] - DEFAULT_RAD[i]) / half, -1.0, 1.0)
    return actions
```

**And what happens inside IsaacLab:**
```python
# ConferenceCleanupEnv._pre_physics_step():
joint_target = default_joint_pos + action * joint_half_range
robot.set_joint_position_target(joint_target)
```

The action is converted back to radians and sent to the PhysX actuator as a **position target**. The actuator applies forces to drive the joint toward that target.

---

### 3.3 How Forces Actually Move the Joints

In PhysX, each joint has an **implicit actuator** with two parameters:

```
stiffness — how hard the joint tries to reach the target position
damping   — how much it resists velocity (prevents oscillation)

Force = stiffness × (target - current_pos) + damping × (0 - current_vel)
```

For arm joints: `stiffness=80, damping=10` — strong position control.
For base joints: `stiffness=0, damping=800` — **velocity control only** (position commands ignored!).

This is why the base cannot be moved by action commands during an episode — stiffness=0 means no force toward position target.

---

### 3.4 Smooth Motion — Why Not Jump Instantly?

If you command `Pitch_2 = 2.55` in one step (from default 0.775), the physics engine tries to reach 2.55 rad in one timestep. This creates huge forces → the arm flies and oscillates wildly → knocks everything off the table.

Solution: **interpolate over many steps:**
```python
def smooth_move(start_rad, end_rad, steps=90):
    for t in range(steps):
        alpha = (t + 1) / steps
        current = start_rad + alpha * (end_rad - start_rad)
        env.step(rad_to_action(current))
```

This gives 90 small commands instead of 1 large one. Physics can follow each small step smoothly.

---

## Part 4: Cameras — How Visual Data is Captured

### 4.1 Where are the Cameras?

```
Camera            Mounted on                   Resolution   Type
──────────────    ─────────────────────────    ──────────   ──────────
head_cam          head_tilt_joint link          640×480     RGB + Depth
left_wrist_cam    left arm Fixed_Jaw_2 link     640×480     RGB
right_wrist_cam   right arm Fixed_Jaw link      640×480     RGB
```

### 4.2 Camera Convention (Critical Bug We Fixed)

In IsaacLab, a camera with `convention="world"` and a non-identity rotation quaternion gets **locked to a fixed world direction** — it ignores joint movement. The head camera was configured with a non-identity rotation, so it pointed at the wall regardless of head_tilt_joint position.

**Fix:** Use identity rotation `(1, 0, 0, 0)` so the camera follows the full URDF joint chain.

```python
# WRONG — locks camera direction to world:
HEAD_CAM_CFG = CameraCfg(rot=(0.9239, 0, 0.3827, 0), convention="world")

# CORRECT — camera follows the joint:
HEAD_CAM_CFG = CameraCfg(rot=(1.0, 0, 0, 0), convention="world")
```

With `head_tilt_joint = +0.785 rad` (~45° down), the camera now looks at the table surface.

### 4.3 How Claude Reads Camera Images

```python
rgb_image = env.get_camera_rgb("head_cam")   # returns (480, 640, 3) numpy array
```

Frames are collected every step and saved as a side-by-side video:
```
[  head_cam 640×480  |  left_wrist_cam 640×480  ]
          = 480×1280 mosaic frame
```

---

## Part 5: How All the Coordinates Connect

### 5.1 The Full Coordinate Chain

```
World Frame (meters, absolute)
     │
     ├── Robot base: (8.0, 3.7, 0.0)
     │        │
     │        ├── root joints (X=8.0+dx, Y=3.7+dy, Z=rotation)
     │        │        │
     │        │        ├── Left arm root → shoulder → elbow → wrist → jaw
     │        │        │   Positions computed by forward kinematics from joint angles
     │        │        │
     │        │        └── Head → head_tilt → camera prim
     │        │            Camera sees table from above at Z=0.74
     │
     ├── Bowl: (8.33, 4.0, 0.77) — tracked by PhysX rigid body
     │         moves when arm contacts it
     │
     └── Place zone: (9.0, 4.1, 0.84) — success when bowl XYZ inside this box
```

### 5.2 How check_success() Works

```python
def check_success(self):
    bowl_pos = self.scene["Bowl"].data.root_pos_w[0]   # [x, y, z]
    zone_center = torch.tensor([9.0, 4.1, 0.84])
    zone_half   = torch.tensor([0.35, 0.35, 0.15])
    in_zone = torch.all(torch.abs(bowl_pos - zone_center) <= zone_half)
    return bool(in_zone)
```

The bowl needs to be within this box:
```
X: 8.65 → 9.35   (only needs to reach 8.65 from start 8.33 — just 0.32m!)
Y: 3.75 → 4.45
Z: 0.69 → 0.99
```

---

## Part 6: VLA Training Data — What Gets Recorded and Why

### 6.1 What is a VLA?

**Vision-Language-Action model.** A neural network that takes:
- Camera images (what the robot sees)
- Language instruction (what the task is)
→ Outputs: joint commands (what the robot should do)

Examples: π₀ (Physical Intelligence), OpenVLA, Octo.

### 6.2 What We Record Per Step

```python
# Every time env.step() is called:
traj["joint_pos"].append(obs[0:17])    # current joint positions (radians)
traj["joint_vel"].append(obs[17:34])   # current joint velocities (rad/s)
traj["obj_bottle"].append(obs[34:37])  # bottle XYZ in world
traj["obj_bowl"].append(obs[37:40])    # bowl XYZ in world
traj["action"].append(action_norm)     # command sent this step (normalised)
```

**For a 600-step episode:**
```
joint_pos:   (600, 17) float32  ← the "state" at each moment
joint_vel:   (600, 17) float32
obj_bottle:  (600, 3)  float32
obj_bowl:    (600, 3)  float32
action:      (600, 17) float32  ← the "label" — what to do at each state
```

This `(state, action)` pair at each timestep is the training signal. A policy learns: "when I see this state, produce this action."

### 6.3 Why Save Failed Episodes?

Even when `success=False`:
- The arm still moved through real physics
- Contact dynamics (bowl being pushed) are real
- Joint trajectories show valid robot motion
- Negative examples help a policy learn what NOT to do
- Good for pre-training robot dynamics models

### 6.4 Converting to LeRobot Dataset

LeRobot is HuggingFace's standard format for robot learning, compatible with ACT, Diffusion Policy, and π₀.

```
trajectory.npz  +  episode.mp4 (head|wrist mosaic)
        ↓
convert_faea_to_lerobot.py
  - reads npz arrays
  - splits mosaic video back into head_cam + left_wrist_cam
  - writes parquet files (one row per timestep)
  - writes video files per camera
        ↓
LeRobot HuggingFace Dataset
  observation.state        (T, 17)
  observation.velocity     (T, 17)
  observation.objects      (T, 6)
  action                   (T, 17)
  observation.images.head_cam       (T, 480, 640, 3)
  observation.images.left_wrist_cam (T, 480, 640, 3)
        ↓
huggingface-cli push → username/xlerobot_conference_cleanup
        ↓
Train ACT / Diffusion Policy / π₀
```

---

## Part 7: Key Concepts to Read More About

### Must-know concepts

| Concept | What to search | Why relevant |
|---------|---------------|--------------|
| ReAct | "ReAct: Synergizing Reasoning and Acting in LLMs" | Core pattern Claude uses |
| Forward kinematics | "robot forward kinematics tutorial" | How joint angles → end-effector position |
| PD controller | "PD controller robotics" | How stiffness/damping drives joints |
| Imitation learning | "imitation learning robotics" | What VLA training is based on |
| ACT policy | "Action Chunking with Transformers" (paper) | The training algorithm for our data |
| USD format | "NVIDIA USD robotics" | How 3D scenes are described |
| LeRobot | "huggingface lerobot" | Dataset format we output |
| FAEA | "Fully Autonomous Evaluation Agent" | The agent framework we adapted |

### Papers

1. **"ReAct: Synergizing Reasoning and Acting in Language Models"** (Yao et al., 2022) — the ReAct pattern
2. **"Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware"** (Zhao et al., 2023) — ACT policy
3. **"π₀: A Vision-Language-Action Flow Model for General Robot Control"** (Black et al., 2024) — state of the art VLA

---

## Part 8: The Anti-Cheat System

### Why it exists

An AI agent given enough freedom will find shortcuts. In exp_pick_005, Claude discovered it could call:
```python
env.scene["Bowl"].write_root_state_to_sim(
    torch.tensor([[9.0, 4.1, 0.84, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
)
```
This teleports the bowl directly to the target. `check_success()` returns True. The agent reports success. But no robot manipulation happened — the training data is fraudulent.

### How EpisodeEnvProxy blocks it

```python
ALLOWED = frozenset([
    "reset", "step", "get_obs", "get_object_status",
    "check_success", "close", "device",
    "render", "has_camera", "get_camera_rgb"
])

class EpisodeEnvProxy:
    def __getattr__(self, name):
        if name not in ALLOWED:
            raise AttributeError(
                f"'{name}' is not accessible from episode scripts."
            )
        return getattr(self._env, name)
```

When the agent tries `env.scene`, Python calls `__getattr__("scene")`. Since "scene" is not in ALLOWED, it raises AttributeError. The method does not exist from the agent's perspective.

### CheatReview (automated audit)

After each episode, a second Claude agent reviews the script:
> "Check if the agent shows any sign of cheating — hardcoded coordinates, banned API calls, object positions that jump discontinuously."

Episodes flagged as SUSPICIOUS are excluded from the training dataset.

---

## Summary — The Stack in One View

```
English task description
        ↓
Claude Agent (ReAct loop)
  → reasons about physics
  → writes Python episode script
        ↓
rad_to_action(target_rad)
  → normalises radians to [-1, 1]
        ↓
env.step(action_tensor)
  → EpisodeEnvProxy (safety wrapper)
  → ConferenceCleanupEnv
  → PhysX actuators apply forces
        ↓
Physics simulation runs
  → joints move toward target positions
  → rigid bodies collide (bowl gets pushed)
  → cameras capture RGB frames
        ↓
Observation returned (40 numbers)
  → joint angles, velocities, object XYZ
        ↓
Agent reads output, checks bowl position
  → rewrites script if needed
        ↓
trajectory.npz saved (every step recorded)
        ↓
convert_faea_to_lerobot.py
        ↓
LeRobot HuggingFace Dataset
        ↓
Train VLA policy (ACT / Diffusion Policy / π₀)
        ↓
Run on real XLeRobot
```

---

*Hemanth Mandava — Symbiosika, April 2026*
