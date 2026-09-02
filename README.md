# Robotic Arm RL

Learning reinforcement learning for robot control using MuJoCo + Gymnasium + Stable-Baselines3.

Two phases:
- `phase1-2dof-arm/` - simple 2-link planar arm, done
- `phase2-6dof-arm/` - UR5e (real 6-DOF industrial arm), in progress

## Phase 1: 2-link arm

Basic setup: 2 links, 2 hinge joints, reaching for a fixed target. Used this to get the pipeline working before trying something harder.

Stages:
1. Load the model, check it works
2. Manual control (sine wave on the joints)
3. Read joint state, compute end-effector position
4. Closed-form IK + PD controller as a baseline
5. PyTorch basics
6. Custom gym env, train PPO
7. Portfolio writeup (this)

### The reward bug

First PPO run (100k steps) looked fine on paper - avg reward went from -68 to -28 - but when I actually watched it, the arm would get close to the target and then wander off before the episode ended. Turned out the reward only cared about current distance, so there was nothing stopping it from drifting once it got close.

Fixed by adding a small penalty for moving fast once it's within 0.1 of the target. Retrained for 300k steps.

After the fix: reliably reaches and holds the target across different starting positions, final distance around 0.015-0.02 (right where the episode is supposed to terminate). One run still had some jitter before settling, so it's not perfect, but it's a lot better than before.

Files in `phase1-2dof-arm/`:
- `arm_model.xml` - the MJCF model
- `test_install.py`, `stage1-3/5_*.py` - build-up stages
- `arm_env.py` - the gym environment
- `train_ppo.py` / `evaluate_trained.py` - training and eval
- `ppo_arm_reach*.zip` - saved models (baseline, with fix, final)

Python 3.11.9, mujoco 3.12.0, gymnasium 1.3.0, sb3 2.9.0, torch 2.13.0. Trained on CPU.

## Phase 2: UR5e (in progress)

Using the UR5e model from MuJoCo Menagerie instead of building one by hand - it's a real 6-DOF arm, well-tested, and gives me something more realistic to work with than the 2-link version.

Main differences from Phase 1:
- Actuators here are position-controlled (you send a target angle, MuJoCo handles the PD internally), not raw torque like Phase 1
- No closed-form IK for 6 joints - will need numerical/Jacobian-based IK instead
- Bigger obs/action space (6 joints instead of 2)

Progress so far: model loads, moves correctly under manual control, forward kinematics confirmed via `mj_kinematics`. Next: numerical IK, then the RL environment.

Model credit: [UR5e MJCF from MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie), BSD-3-Clause license, original description by Universal Robots.
