"""Microduck *polite bow* task — the simplest episodic trick in the family.

The robot starts in the walking default pose (HOME/STAND2), bows its HEAD
forward to a polite angle, holds it briefly, and brings it back up to standing.
The head never goes near the ground (that is ground_pick's job) and the feet
never leave the floor.

Phase encoding (twist slot, 3-D), same contract as ground_pick / roller_crouch:
    command = [cos(2π·phase), sin(2π·phase), 0]
so the trick rides the runtime's ground-pick slot and publishes as
``--kind episodic --duration-s 4.0``.

── Design ────────────────────────────────────────────────────────────────────
Built on ``make_microduck_velocity_env_cfg`` (NOT on mjlab's base template):
that inherits the whole sim2real stack — DR, obs noise, sensor delays, IMU
misalignment, encoder bias, BAM friction expansion, the NaN guard — already in
sync with the walking policy. What this file does is swap the objective:

  - twist becomes a phase signal (GroundPickPhaseCommand),
  - the walking reward stack is removed,
  - the main reward is ``phase_pose_track`` on the four neck/head joints
    against a target interpolated HOME ↔ BOW_POSE along the phase profile
    (roller_crouch's recipe). Coming back UP is rewarded exactly as going
    down — symmetric by construction, no jackpot at the bottom.
  - legs are held near HOME by their own Gaussian, loose enough (std 0.25) that
    the policy can counterbalance the head with hips/ankles.

The head is the task here, so the head_pose / body_pose command slots are
ZERO-PADDED (like ground_pick) rather than tracked — the 61D obs layout is
unchanged so the runtime can hot-swap this policy with the others.

── Measured before training (2026-09, scratchpad sweeps) ─────────────────────
  - Sign conventions: neck_pitch DECREASING swings the head forward+down;
    head_pitch INCREASING tips the beak down. head_yaw/head_roll stay at 0 so
    the bow is purely sagittal.
  - Static balance: the head is ~38% of body mass, so bowing it forward moves
    the whole-body CoM forward by 26 mm (from +0.6 mm at HOME). The foot
    support polygon is 71 mm fore/aft, leaving ~17 mm of forward margin at the
    chosen depth — feasible without a prescribed trunk lean, so none is
    prescribed and the policy finds its own counterbalance. Run 1 confirmed
    this: zero falls and full-length episodes from ~iteration 150 onward.
  - Depth is set by the keyframe, not by training. Run 1 tracked the commanded
    target to within 1.3° (bow_pose 5.90/6.00) while still looking shallow —
    the reward had nothing left to push on. If the bow looks wrong, change
    BOW_POSE; do not train longer.
  - A settle test is NOT meaningful for this task: microduck cannot hold ANY
    standing pose open-loop (XML kp 0.55; it tips over from the STAND keyframe
    in ~1.5 s with BAM actuators). Balance here is the policy's job, and the
    static CoM margin above is the check that replaces it.

Joint layout (14 actuated joints):
    0-4 : left  leg (hip_yaw, hip_roll, hip_pitch, knee, ankle)
    5-8 : neck/head (neck_pitch, head_pitch, head_yaw, head_roll)
    9-13: right leg (hip_yaw, hip_roll, hip_pitch, knee, ankle)
"""

import dataclasses
from copy import deepcopy

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import (
    CurriculumTermCfg,
    EventTermCfg,
    ObservationTermCfg,
    RewardTermCfg,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg
from mjlab.tasks.velocity import mdp

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    NUM_STEPS_PER_ENV,
    make_microduck_velocity_env_cfg,
)
from mjlab_microduck.tasks.symmetry import PpoWithSymmetryCfg, SYMMETRY_CFG

# Symmetry — off, like every v1.5 env (SYMMETRY_CFG's permutation table predates
# the 61D obs layout).
ENABLE_SYMMETRY = False

# ── Phase profile ─────────────────────────────────────────────────────────────
# 4 segments over a 4 s period:
#   descend [0, DESCENT_END)        1.20 s   HOME -> BOW
#   hold    [DESCENT_END, HOLD_END) 0.40 s   held at the bow
#   rise    [HOLD_END, RISE_END)    0.88 s   BOW -> HOME
#   rest    [RISE_END, 1.0)         1.52 s   standing
# ⚠️ RISE_END must stay BELOW 0.7: scripts/infer_policy.py hands control back to
# the walking policy at φ ≥ 0.7 (see update_ground_pick_phase). ground_pick has
# a ⚠️ in its header for busting that window — this profile finishes the rise
# with room to spare. BOW_PERIOD must match --ground-pick-period at deployment
# and duration_s in the published manifest.
BOW_PERIOD  = 4.0
DESCENT_END = 0.30
HOLD_END    = 0.40
RISE_END    = 0.62

# ── BOW keyframe (rad, by joint NAME — resolution by name keeps this correct on
# the backlash model, where passive joints interleave). Offsets from HOME
# (neck_pitch/head_pitch = +0.3491):
#     neck_pitch  HOME - 1.00  → head swings forward and down
#     head_pitch  HOME + 0.60  → beak tips to exactly vertical
# Mouth ends 93 mm below its standing height, ~159 mm above the floor: an
# unmistakable deep bow, still nowhere near the ground.
#
# This is the DEEPEST sensible bow, not merely a deep one. Swept against the
# 71 mm fore/aft support polygon (head drop | beak angle | CoM front margin):
#     -0.70 / +0.40   67 mm |  63° | 20.8 mm   ← run 1: read as too shallow
#     -0.85 / +0.50   81 mm |  78° | 18.2 mm
#     -1.00 / +0.60   93 mm |  90° | 16.6 mm   ← here
#     -1.20 / +0.70  104 mm |  70° | 15.8 mm
#     -1.40 / +0.80  110 mm |  53° | 16.4 mm
# Depth is nearly free in balance terms — 39% deeper costs 4 mm of margin,
# because past this point the head tucks back toward the body and the CoM stops
# travelling forward. But the BEAK ANGLE peaks here at vertical and then falls:
# beyond -1.00/+0.60 the head rotates past vertical and the beak points back
# under the robot, which reads as the head curling under rather than a deeper
# bow. Going further looks worse, not deeper.
#
# ONLY the two joints that actually move belong here. head_yaw / head_roll are
# held at HOME by `bow_sagittal` below instead. They used to sit in this dict at
# 0.0, and because phase_pose_track averages the Gaussian over its joints, two
# always-perfect joints contributed 1.0 every step no matter what the policy did
# — diluting the signal by half. Measured over a full phase cycle:
#     4 joints (yaw/roll pinned here): stand-still 4.55 vs perfect 6.00 → signal 1.45
#     2 joints (this version):         stand-still 3.09 vs perfect 6.00 → signal 2.91
# i.e. doing nothing scored 76% of the maximum. Keep this dict to moving joints.
BOW_POSE = {
    "neck_pitch": 0.3491 - 1.00,
    "head_pitch": 0.3491 + 0.60,
}
# Tracking std ≈ the error we still care about (~9°), not the max error.
BOW_POSE_STD = 0.15
# Legs: loose enough to let the policy counterbalance the head, tight enough
# that "stand still" stays the stance.
LEG_POSE_STD = 0.25
# head_yaw / head_roll: tighter than the legs — the legs need freedom to
# counterbalance, a bow does not need to twist.
SAGITTAL_STD = 0.15

_LEG_JOINTS = [0, 1, 2, 3, 4, 9, 10, 11, 12, 13]
# head_yaw, head_roll (servo indices; see the joint layout in the docstring).
_YAW_ROLL_JOINTS = [7, 8]

# Pushes: the bow is a quasi-static gesture, so pushes start gentle and ramp in
# only after the motion exists (ground_pick's ±0.3 "made it fall even standing
# straight"; the sit env unlearned its transition when pushed too early).
PUSH_RANGE_INITIAL = (-0.10, 0.10)
PUSH_RANGE_FINAL   = (-0.25, 0.25)
PUSH_INTERVAL_S    = (3.0, 6.0)

# Walking-only reward terms with no meaning for a standing bow.
_WALKING_REWARDS = (
    "track_linear_velocity",
    "track_angular_velocity",
    "air_time",
    "foot_clearance",
    "foot_swing_height",
    "foot_slip",
    "pose",                 # replaced by leg_stance (the base term keys its std
                            # off command speed, which is constant 1.0 here)
    "head_pose_tracking",   # the head is the TASK, not a command to follow
    "head_pose_bias",
    "body_pose_tracking",
)

# Curriculum terms that belong to the walking objective.
_WALKING_CURRICULA = (
    "standing_envs",
    "head_pose_range",
    "body_pose_range",
    "head_pose_bias_weight",
)


def make_microduck_polite_bow_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create the Microduck polite-bow environment configuration (flat only)."""

    # Flat terrain only: a bow is a standing gesture — rough terrain would only
    # confound it. Inherits the walk robot model (feet-only collision), which is
    # what the head-never-touches-the-floor assumption allows.
    cfg = make_microduck_velocity_env_cfg(play=play, rough=False)

    # ── Rewards: drop the walking stack ───────────────────────────────────────
    for name in _WALKING_REWARDS:
        cfg.rewards.pop(name, None)

    # ── Rewards: the bow itself ───────────────────────────────────────────────
    # Pose interpolated HOME ↔ BOW along the phase blend. Directive and
    # symmetric: at every instant it names the neck/head configuration wanted,
    # and the rise is paid exactly like the descent, so there is no jackpot for
    # arriving at the bottom early.
    _bow_params = {
        "command_name": "twist",
        "target_pose": BOW_POSE,
        "descent_end": DESCENT_END,
        "hold_end": HOLD_END,
        "rise_end": RISE_END,
    }
    cfg.rewards["bow_pose"] = RewardTermCfg(
        func=microduck_mdp.phase_pose_track,
        weight=6.0,
        params={**_bow_params, "std": BOW_POSE_STD},
    )
    # L1 bootstrap: constant gradient toward the target even where the Gaussian
    # has saturated near zero (i.e. before the motion exists at all).
    cfg.rewards["bow_pose_l1"] = RewardTermCfg(
        func=microduck_mdp.phase_pose_track_l1,
        weight=2.0,
        params=_bow_params,
    )

    # Legs hold the standing stance throughout (single fixed HOME target, no
    # phase gating — the legs are not part of the gesture).
    cfg.rewards["leg_stance"] = RewardTermCfg(
        func=microduck_mdp.pose_target_match,
        weight=2.0,
        params={"std": LEG_POSE_STD, "joint_indices": _LEG_JOINTS},
    )

    # Keep the bow sagittal: head_yaw / head_roll held at HOME. Separate from
    # bow_pose on purpose — folding them into the tracked target halves that
    # reward's gradient (see the BOW_POSE comment), and they want a tighter
    # tolerance than the legs anyway.
    cfg.rewards["bow_sagittal"] = RewardTermCfg(
        func=microduck_mdp.pose_target_match,
        weight=1.0,
        params={"std": SAGITTAL_STD, "joint_indices": _YAW_ROLL_JOINTS},
    )

    # Both feet stay planted and flat: the bow must not turn into a step, a
    # lunge, or a roll onto the foot edge.
    cfg.rewards["feet_grounded"] = RewardTermCfg(
        func=microduck_mdp.feet_grounded_reward,
        weight=2.0,
        params={"sensor_name": "feet_ground_contact"},
    )
    cfg.rewards["feet_flat"] = RewardTermCfg(
        func=microduck_mdp.feet_flat_penalty,
        weight=-2.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", site_names=["left_foot", "right_foot"]
            ),
        },
    )

    # ── Rewards: regularisation ───────────────────────────────────────────────
    # upright / body_ang_vel / angular_momentum / self_collisions / dof_pos_limits
    # are inherited from the velocity env unchanged: the trunk should stay
    # vertical (this is a head bow, not a trunk bow), and the motion is slow
    # enough that the motion-blocker concern does not apply.
    #
    # action_rate_l2 and neck_action_rate_l2 are both ramped by curriculum below
    # rather than set flat: an attempt-tax on the neck while the bow is still
    # being discovered makes "keep the head still" win.
    cfg.rewards["action_rate_l2"].weight = -0.3
    cfg.rewards["neck_action_rate_l2"] = RewardTermCfg(
        func=microduck_mdp.neck_action_rate_l2, weight=0.0
    )
    cfg.rewards["joint_torques_l2"] = RewardTermCfg(
        func=microduck_mdp.joint_torques_l2, weight=-1e-3
    )

    # ── Command: the walking twist becomes a phase signal ─────────────────────
    # GroundPickPhaseCommandCfg is a UniformVelocityCommandCfg; the inherited
    # twist is a VelocityCommandCommandOnlyCfg with extra walking-only fields
    # (rel_turn_in_place_envs), so copy across only the fields it declares.
    _phase_fields = {f.name for f in dataclasses.fields(microduck_mdp.GroundPickPhaseCommandCfg)}
    _twist_kwargs = {
        k: deepcopy(v)
        for k, v in vars(cfg.commands["twist"]).items()
        if k in _phase_fields
    }
    _twist_kwargs.update(
        class_type=microduck_mdp.GroundPickPhaseCommand,
        period=BOW_PERIOD,
        # Random start phase per episode decorrelates envs. The runtime always
        # triggers at φ=0 from standing, which the rest segment covers.
        randomize_phase=True,
        rel_standing_envs=0.0,
        rel_heading_envs=0.0,
    )
    cfg.commands["twist"] = microduck_mdp.GroundPickPhaseCommandCfg(**_twist_kwargs)

    # ── Commands / obs: zero-pad the head + body pose slots ───────────────────
    # The head is driven by the task's phase motion, so tracking a head command
    # too would fight it. The slots stay in the obs (constant zero) to keep the
    # unified 61D layout the runtime feeds every policy.
    cfg.commands.pop("head_pose", None)
    cfg.commands.pop("body_pose", None)
    for group in ("actor", "critic"):
        cfg.observations[group].terms["head_command"] = ObservationTermCfg(
            func=microduck_mdp.zero_command_padding, params={"dim": 4},
        )
        cfg.observations[group].terms["body_command"] = ObservationTermCfg(
            func=microduck_mdp.zero_command_padding, params={"dim": 6},
        )

    # ── Events ────────────────────────────────────────────────────────────────
    # Everything else (BAM friction expansion, action-history reset, foot
    # friction, CoM / mass-inertia / joint-friction / armature DR, encoder bias)
    # is inherited from the velocity env unchanged.
    if "push_robot" in cfg.events:
        cfg.events["push_robot"] = EventTermCfg(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            # Play: spaced pushes, to judge the gesture on realistic behaviour
            # rather than under fire (the velocity env's play interval is a
            # 0.5-1.0 s stress test).
            interval_range_s=(2.0, 4.0) if play else PUSH_INTERVAL_S,
            params={
                "velocity_range": {
                    "x": PUSH_RANGE_INITIAL,
                    "y": PUSH_RANGE_INITIAL,
                },
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )

    # ── Curriculum ────────────────────────────────────────────────────────────
    for name in _WALKING_CURRICULA:
        cfg.curriculum.pop(name, None)

    # Smoothness AFTER discovery. The bow is a slow, careful motion, so it ends
    # up heavier than walking's -1.0 (ground_pick's lesson), but it starts light
    # so the gross motion can form first.
    cfg.curriculum["action_rate_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "action_rate_l2",
            "weight_stages": [
                {"step": 0,                         "weight": -0.3},
                {"step": 200 * NUM_STEPS_PER_ENV,   "weight": -0.8},
                {"step": 400 * NUM_STEPS_PER_ENV,   "weight": -1.5},
                {"step": 600 * NUM_STEPS_PER_ENV,   "weight": -2.0},
            ],
        },
    )
    # The neck carries the whole gesture, so its own smoothness term stays at 0
    # until the bow exists, then damps the jitter.
    cfg.curriculum["neck_action_rate_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "neck_action_rate_l2",
            "weight_stages": [
                {"step": 0,                         "weight": 0.0},
                {"step": 300 * NUM_STEPS_PER_ENV,   "weight": -0.2},
                {"step": 600 * NUM_STEPS_PER_ENV,   "weight": -0.5},
            ],
        },
    )
    # Pushes ramp in only once the bow is consolidated.
    cfg.curriculum["push_magnitude"] = CurriculumTermCfg(
        func=microduck_mdp.push_curriculum,
        params={
            "event_name": "push_robot",
            "push_stages": [
                {"step": 0,
                 "velocity_range": {"x": PUSH_RANGE_INITIAL, "y": PUSH_RANGE_INITIAL}},
                {"step": 400 * NUM_STEPS_PER_ENV,
                 "velocity_range": {"x": (-0.18, 0.18), "y": (-0.18, 0.18)}},
                {"step": 700 * NUM_STEPS_PER_ENV,
                 "velocity_range": {"x": PUSH_RANGE_FINAL, "y": PUSH_RANGE_FINAL}},
            ],
        },
    )

    return cfg


# ── RL runner config ──────────────────────────────────────────────────────────

MicroduckPoliteBowRlCfg = RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,  # baked into the ONNX by scripts/export.py
        distribution_cfg={
            "class_name": "GaussianDistribution",
            "init_std": 1.0,
            "std_type": "scalar",
        },
    ),
    critic=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
    ),
    algorithm=PpoWithSymmetryCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        symmetry_cfg=SYMMETRY_CFG if ENABLE_SYMMETRY else None,
    ),
    wandb_project="mjlab_microduck",
    experiment_name="polite_bow",
    run_name="polite_bow",
    save_interval=250,
    num_steps_per_env=NUM_STEPS_PER_ENV,
    max_iterations=20_000,
)
