Good. Now we can talk about an actual porting plan instead of a conceptual one.

The first thing to understand is:

```text
HoST
├── rsl_rl
└── legged_gym
```

In MJLab, **you will throw away almost the entire `rsl_rl` folder**.

MJLab already has:

```text
src/mjlab/agents/rsl_rl/
```

which replaces:

```text
HoST/rsl_rl/
```

So:

```text
HoST/rsl_rl/*
                X DO NOT PORT
```

You only port environment logic.

---

# STEP 1: Ignore rsl_rl completely

Do not touch:

```text
rsl_rl/algorithms/
rsl_rl/modules/
rsl_rl/runners/
rsl_rl/storage/
rsl_rl/env/
```

Equivalent already exists in MJLab.

| HoST                | MJLab                 |
| ------------------- | --------------------- |
| actor_critic.py     | MJLab RSL-RL          |
| ppo.py              | MJLab PPO             |
| rollout_storage.py  | MJLab rollout storage |
| on_policy_runner.py | MJLab runner          |

Port effort:

```text
0 files
```

---

# STEP 2: Port robot asset

HoST:

```text
resources/robots/g1/
    g1_23dof.urdf
    meshes/
```

Check whether MJLab already has G1.

If MJLab already contains:

```text
src/mjlab/assets/unitree/g1/
```

then:

```text
PORT NOTHING
```

Use MJLab's robot.

This is what I recommend.

---

# STEP 3: Find the actual environment

The core files are:

```text
host_ground.py
host_ground_prone.py
host_platform.py
host_slope.py
host_wall.py
```

These are the files that matter.

Everything else configures them.

---

# STEP 4: Create MJLab task folder

Create:

```text
src/mjlab/tasks/standing/g1_host/
```

Inside:

```text
src/mjlab/tasks/standing/g1_host/
├── __init__.py
├── g1_host_env_cfg.py
├── g1_host_rewards.py
├── g1_host_events.py
├── g1_host_observations.py
└── g1_host_agent_cfg.py
```

---

# STEP 5: Port g1_config_ground.py

HoST:

```text
envs/g1/g1_config_ground.py
```

contains:

```python
class G1Cfg(...)
class G1CfgPPO(...)
```

These become:

```text
g1_host_env_cfg.py
g1_host_agent_cfg.py
```

Mapping:

| HoST     | MJLab     |
| -------- | --------- |
| G1Cfg    | EnvCfg    |
| G1CfgPPO | RunnerCfg |

---

# STEP 6: Port observations

Look inside:

```text
host_ground.py
```

Find:

```python
def compute_observations()
```

Everything in there becomes:

```text
g1_host_observations.py
```

Example:

HoST:

```python
obs = torch.cat([
    base_ang_vel,
    projected_gravity,
    dof_pos,
    dof_vel,
    actions
], dim=-1)
```

MJLab:

```python
ObsTerm(
    func=mdp.base_ang_vel
)

ObsTerm(
    func=mdp.projected_gravity
)

ObsTerm(
    func=mdp.joint_pos_rel
)

ObsTerm(
    func=mdp.joint_vel_rel
)

ObsTerm(
    func=mdp.last_action
)
```

This is usually 1-2 hours.

---

# STEP 7: Port rewards

This is the most important file.

Search:

```python
def compute_reward()
```

inside:

```text
host_ground.py
host_ground_prone.py
```

You'll find terms like:

```python
reward_upright
reward_height
reward_joint_regularization
reward_torque
reward_action_rate
```

Create:

```text
g1_host_rewards.py
```

Each reward becomes:

```python
def upright_reward(...)
```

then register:

```python
RewardTermCfg(
    func=upright_reward,
    weight=...
)
```

inside:

```python
self.rewards
```

of

```text
g1_host_env_cfg.py
```

---

# STEP 8: Port reset logic

Most important file:

```text
host_ground_prone.py
```

Look for:

```python
def reset_idx()
```

This is where HoST generates:

```text
supine
prone
side
kneeling
```

states.

Port these into:

```text
g1_host_events.py
```

Example:

HoST:

```python
root_state[:, 3:7] = quat
dof_pos[:] = pose
```

MJLab:

```python
EventTermCfg(
    func=random_prone_reset
)
```

This is where most of the standing-up behavior comes from.

---

# STEP 9: Port curriculum

Search:

```python
curriculum
```

or

```python
update_curriculum()
```

inside:

```text
host_ground_prone.py
```

Create:

```text
g1_host_curriculum.py
```

or embed inside env cfg.

Map:

```text
easy pose
↓
hard pose
↓
full prone pose
```

curriculum.

---

# STEP 10: Ignore scripts

Do NOT port:

```text
scripts/train.py
scripts/play.py
scripts/eval/*
```

MJLab already has:

```bash
uv run train ...
uv run play ...
```

---

# Actual minimal port

If your goal is:

```text
fallen state
      ↓
neutral stand
```

you only need:

```text
host_ground_prone.py
g1_config_ground_prone.py
g1_utils.py
```

Everything else can be ignored initially.

So the real mapping is:

```text
HoST
│
├── host_ground_prone.py
│        ↓
│   g1_host_rewards.py
│   g1_host_events.py
│   g1_host_observations.py
│
├── g1_config_ground_prone.py
│        ↓
│   g1_host_env_cfg.py
│   g1_host_agent_cfg.py
│
└── g1_utils.py
         ↓
    utility functions
```

That is probably **90% of the useful code** for reproducing HoST in MJLab.

My recommendation would be to start by opening only these three files:

```text
host_ground_prone.py
g1_config_ground_prone.py
g1_utils.py
```

and ignore the other 95% of the repository. Those three files define almost all of the standing-up task logic you actually care about.
