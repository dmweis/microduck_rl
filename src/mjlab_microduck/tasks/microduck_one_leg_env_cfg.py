"""Microduck *one-leg balance* task — lift one foot, hold, put it back down.

The robot starts in the walking stance, shifts its weight onto the SUPPORT foot
(``SUPPORT_FOOT`` below), lifts the other knee into a flamingo, holds it for
~2 s, then lowers back to a clean two-footed stand.

Phase encoding (twist slot, 3-D), same contract as polite_bow / ground_pick:
    command = [cos(2π·phase), sin(2π·phase), 0]
so the trick rides the runtime's ground-pick slot and publishes as
``--kind episodic --duration-s 6.0``.

── Measured BEFORE training — read this before touching the rewards ──────────
This task is at the edge of what the hardware can do, and every unusual choice
below follows from numbers, not taste. All figures are whole-body CoM expressed
in the support foot's site frame (its Z+ is the sole normal, so that frame's XY
plane IS the ground); FK sweeps on ``robot_walk.xml``, 2026-09.

  - **The problem is lateral, and it is big.** The feet are 83.6 mm apart and a
    sole is 47 × 29 mm. At HOME the CoM sits 44.7 mm OUTSIDE the support sole:
    a one-leg stand has to move it ~45 mm sideways.

  - **There is no ankle-roll DOF.** Trunk roll relative to a planted foot is
    exactly ``-hip_roll``, so ALL lateral authority lives in the support hip:
    roll (38 mm across its full ±22°) and yaw (25 mm). Yaw matters for a
    non-obvious reason — it turns the body so the CoM offset points down the
    sole's LONG axis (47 mm) instead of its short one (29 mm).

  - **Best achievable static margin** (CoM to the nearest sole edge):

        support hips at 90% of range, head neutral      2.8 mm
        support hips at their mechanical stops          7.0 mm
        hips at stops + head cranked as counterweight   9.9 mm

    7.0 mm over a 128 mm-high CoM is ±3.1° of tilt tolerance. It is genuinely
    tight, but not absurd: a human on one leg has ~2.3°.

  - **hip_roll's ±22° is a CAD-derived mechanical limit** (``robot_walk.xml``,
    exported from Onshape), not a software cap. It cannot be tuned away, and it
    is the single number that decides this task.

  - **Authority at the optimum is ONE-SIDED.** From the best pose, the swing leg
    and head can move the CoM 41.7 mm back toward the midline but only 1.1 mm
    further over the foot. The robot can always fall; it can barely push back.
    That is why the support hip and the head are NOT pinned by a keyframe here
    — they are the balance actuators, and a tight pose reward would confiscate
    exactly the authority the task needs.

  - **The swing leg is limited by self-collision before balance.** The thigh
    lift that best counterweights the CoM also drives the swing shank into the
    trunk battery holder. The shipped-then-fixed keyframe interpenetrated by
    8.07 mm; backing the thigh off to -30° and taking the height back at the
    knee costs 1.0 mm of CoM margin and clears it. Never solve this pose without
    a self-contact constraint — ``test_flamingo_keyframe_is_self_collision_free``
    is the guard, and it must keep passing.

  - A settle test is not meaningful (as for polite_bow): microduck cannot hold
    ANY standing pose open-loop (XML kp 0.55). The static margin above is the
    check that replaces it.

── Design ────────────────────────────────────────────────────────────────────
Built on ``make_microduck_velocity_env_cfg`` so the whole sim2real stack (DR,
obs noise, sensor delays, IMU misalignment, encoder bias, BAM friction
expansion, the NaN guard) stays in sync with the walking policy. What changes:

  - twist becomes a phase signal (GroundPickPhaseCommand, 6 s),
  - the walking reward stack is removed,
  - the main reward is ``com_over_support_phased`` — the CoM over the support
    foot, which is what balancing on one leg literally IS. Not a pose proxy.
  - the SWING leg gets a phase-interpolated flamingo keyframe, deliberately
    LOOSE (std 0.35), and the support leg / head get looser ones still (0.50).
    They exist to define the gesture and to restore a clean stand during the
    rest segment — not to dictate the balance strategy.
  - ``swing_foot_air_phased`` is the hard state gate that makes it actually
    one-legged; ``support_foot_grounded`` is the always-on anti-hop.

The head is a balance actuator here, so the head_pose / body_pose command slots
are ZERO-PADDED (like polite_bow) rather than tracked — the 61D obs layout is
unchanged so the runtime can hot-swap this policy with the others.

Joint layout (14 actuated joints):
    0-4 : left  leg (hip_yaw, hip_roll, hip_pitch, knee, ankle)
    5-8 : neck/head (neck_pitch, head_pitch, head_yaw, head_roll)
    9-13: right leg (hip_yaw, hip_roll, hip_pitch, knee, ankle)
"""

import dataclasses
import math
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
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity import mdp

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    NUM_STEPS_PER_ENV,
    make_microduck_velocity_env_cfg,
)
from mjlab_microduck.tasks.symmetry import PpoWithSymmetryCfg, SYMMETRY_CFG

# ── Support foot: "left" or "right" ───────────────────────────────────────────
# Flips which leg balances and which leg lifts, exactly like ball_kick's
# KICK_FOOT. Train the mirrored policy as a separate run with this flipped;
# no other change is needed (HOME is left/right symmetric and the keyframe
# below is mirrored programmatically).
SUPPORT_FOOT = "left"
assert SUPPORT_FOOT in ("left", "right")

# Symmetry — must stay OFF: standing on one leg is inherently asymmetric.
ENABLE_SYMMETRY = False

# ── Phase profile ─────────────────────────────────────────────────────────────
# 4 segments over a 6 s period:
#   lift  [0, LIFT_END)          0.90 s   HOME -> FLAMINGO
#   hold  [LIFT_END, HOLD_END)   2.10 s   one leg
#   lower [HOLD_END, DOWN_END)   0.72 s   FLAMINGO -> HOME
#   rest  [DOWN_END, 1.0)        2.28 s   standing
# ⚠️ DOWN_END must stay BELOW 0.7: scripts/infer_policy.py hands control back to
# the walking policy at φ ≥ 0.7 (see update_ground_pick_phase). A profile still
# lowering at 0.7 gets cut off and the walking policy inherits a raised leg.
# ONE_LEG_PERIOD must match --ground-pick-period at deployment and duration_s in
# the published manifest.
ONE_LEG_PERIOD = 6.0
LIFT_END = 0.15
HOLD_END = 0.50
DOWN_END = 0.62

# ── FLAMINGO keyframe (rad, by joint NAME — name resolution keeps this correct
# on the backlash model, where passive joints interleave).
#
# Solved, not guessed: maximise the CoM's distance to the nearest support-sole
# edge subject to head_yaw/head_roll neutral (the bird looks where it is going)
# and the swing foot ≥ 60 mm clear. Support hips land at 90% of range — the
# keyframe deliberately stops short of the mechanical stops so the policy keeps
# overshoot headroom in the ONE direction that can save it. Its tracking std is
# loose enough (below) that the policy may go further when it needs to.
#
# Written for a LEFT support foot; _mirror() flips it when SUPPORT_FOOT is right.
_FLAMINGO_LEFT_SUPPORT = {
    # Support leg — the balance posture. Roll takes the weight sideways, yaw
    # turns the body so the CoM offset runs down the sole's long axis.
    "left_hip_yaw":    -0.3927,   # -22.5°  (HOME   0.0°)
    "left_hip_roll":   +0.3368,   # +19.3°  (HOME  -5.0°)
    "left_hip_pitch":  +0.0276,   #  +1.6°  (HOME -26.2°)
    "left_knee":       -0.5344,   # -30.6°  (HOME  -0.3°)
    "left_ankle":      -0.3568,   # -20.4°  (HOME +26.0°)
    # Swing leg — knee up and forward, sole ~57 mm off the floor.
    #
    # ⚠️ The thigh lift is limited by SELF-COLLISION, not by balance. The first
    # version of this keyframe used hip_pitch -64.9° / knee +63.6°, which drove
    # the swing shank 8.07 mm INTO the trunk battery holder (geoms leg_2 and
    # trunk_base — the contype=2 self-collision set on the walk model). The pose
    # solve had constrained CoM margin and foot clearance but never self-contact,
    # so the reward was commanding the robot into its own body and the policy
    # dutifully learned to press its leg there.
    #
    # Measured with the contact solver at an inflated geom_margin (mj_geomDistance
    # returns exactly 0.0 for these mesh-mesh pairs when apart, so it CANNOT tell
    # "far" from "touching" and is useless as a constraint — see the test):
    #
    #   thigh   knee   CoM margin   self-clearance   foot height
    #    -64.9  +63.6      2.82 mm        -8.07 mm       105 mm   ← shipped, INVALID
    #    -45    +63.6      2.37           -1.05           70
    #    -40    +63.6      2.14           +0.04           62
    #    -35    +63.6      1.87           +0.85           54
    #    -30    +75        1.83           +1.38           57      ← here
    #    -30    +85        2.00           +1.38           67      (knee inside the
    #                                                             dof_pos_limits
    #                                                             band, ±83.2°)
    #
    # Self-clearance is governed almost entirely by the THIGH lift; knee flexion
    # barely moves it, but does buy back foot height and CoM margin. So the thigh
    # comes down for clearance and the knee goes up to keep the flamingo legible.
    # Costs 1.0 mm of CoM margin against a pose that was never actually valid.
    "right_hip_yaw":   +0.3927,   # +22.5°  (HOME   0.0°)
    "right_hip_roll":  -0.3368,   # -19.3°  (HOME  +5.0°)
    "right_hip_pitch": -0.5236,   # -30.0°  (HOME +26.2°)  thigh lift, clearance-limited
    "right_knee":      +1.3090,   # +75.0°  (HOME   0.3°)  below the ±83.2° soft limit
    # Head — counterweights the lift; yaw/roll stay OUT of the keyframe on
    # purpose (see HEAD_JOINTS below).
    "neck_pitch":      +0.8297,   # +47.5°  (HOME +20.0°)
    "head_pitch":      -0.2966,   # -17.0°  (HOME +20.0°)
}
# right_ankle is omitted: the solve left it 2% from HOME, and polite_bow's
# lesson is that a joint sitting at its HOME value scores a perfect 1.0 every
# step and dilutes the Gaussian mean over the tracked set. Only moving joints
# belong in a tracked target.


def _mirror_pose(pose: dict) -> dict:
    """Mirror a left-support keyframe into a right-support one.

    HOME is left/right antisymmetric (roll/pitch/knee/ankle flip sign, yaw
    flips), so mirroring is: swap the leg prefixes and negate. Head joints:
    neck_pitch/head_pitch are sagittal and unchanged.
    """
    out = {}
    for name, val in pose.items():
        if name.startswith("left_"):
            out["right_" + name[len("left_"):]] = -val
        elif name.startswith("right_"):
            out["left_" + name[len("right_"):]] = -val
        else:
            out[name] = val
    return out


FLAMINGO_POSE = (
    _FLAMINGO_LEFT_SUPPORT
    if SUPPORT_FOOT == "left"
    else _mirror_pose(_FLAMINGO_LEFT_SUPPORT)
)

# ── Tracking tolerances ───────────────────────────────────────────────────────
# These are the load-bearing numbers of the whole design. The measured authority
# at the balance optimum is one-sided (41.7 mm back toward the midline, 1.1 mm
# further over the foot), so the swing leg and head are the ONLY correction
# levers the policy has left. Tight pose tracking on them would price away the
# balancing itself — the AGENTS rule "price only the escapable part".
SWING_POSE_STD   = 0.35   # ~20°: names the flamingo, does not dictate it
SUPPORT_POSE_STD = 0.50   # ~29°: a suggestion; the CoM reward is the real boss
HEAD_POSE_STD    = 0.50   # ~29°: the head must stay free to counterweight

# CoM ↔ support-foot Gaussian. 0.025 m ≈ the sole's lateral half-width, i.e. the
# error we still care about (AGENTS: std ≈ that, not the max error).
COM_SUPPORT_STD = 0.025

_LEFT_LEG  = ["left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle"]
_RIGHT_LEG = ["right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle"]
_HEAD_SAGITTAL = ["neck_pitch", "head_pitch"]

# head_yaw / head_roll are deliberately UNPRICED — no keyframe, no HOME term.
# They are worth ~3 mm of static margin (9.9 vs 7.0), which is 40% more margin,
# and the policy should be free to discover that counterweight. action_rate and
# joint_torques are the only things damping them.

# ── Domain randomisation caps — this task cannot take the walking ranges ──────
# The static margin is ≤ 7 mm. The velocity recipe's CoM curriculum ramps trunk
# CoM DR to ±15 mm and head CoM to ±10 mm; either alone exceeds the entire
# margin, so a large share of envs would be kinematically unable to balance and
# the policy would be trained against noise. Capped here at ±5 mm, well inside
# the margin. This IS a sim2real concession and it is deliberate: a task the
# robot cannot do in a given env teaches nothing.
COM_DR_CAP      = 0.005
HEAD_COM_DR_CAP = 0.005

# Pushes: ±0.3 m/s (the walking value) topples a one-leg stand outright. Start
# at zero and ramp only after the balance exists.
PUSH_RANGE_INITIAL = (-0.03, 0.03)
PUSH_RANGE_FINAL   = (-0.10, 0.10)
PUSH_INTERVAL_S    = (4.0, 8.0)

# Walking-only reward terms with no meaning for a one-leg stand.
_WALKING_REWARDS = (
    "track_linear_velocity",
    "track_angular_velocity",
    "air_time",
    "foot_clearance",
    "foot_swing_height",
    "foot_slip",
    "pose",                 # gait-conditioned std; replaced by the phase targets
    "head_pose_tracking",   # the head is a balance actuator, not a command
    "head_pose_bias",
    "body_pose_tracking",
)

_WALKING_CURRICULA = (
    "standing_envs",
    "head_pose_range",
    "body_pose_range",
    "head_pose_bias_weight",
)


def make_microduck_one_leg_env_cfg(
    play: bool = False,
    support_foot: str | None = None,
) -> ManagerBasedRlEnvCfg:
    """Create the Microduck one-leg-balance environment configuration (flat only).

    ``support_foot`` overrides the module-level SUPPORT_FOOT flag (used by
    tests); normal training just sets the flag at the top of this file.
    """
    support_foot = support_foot or SUPPORT_FOOT
    assert support_foot in ("left", "right")
    swing_foot = "right" if support_foot == "left" else "left"
    flamingo = (
        _FLAMINGO_LEFT_SUPPORT
        if support_foot == "left"
        else _mirror_pose(_FLAMINGO_LEFT_SUPPORT)
    )
    support_leg = _LEFT_LEG if support_foot == "left" else _RIGHT_LEG
    swing_leg = _RIGHT_LEG if support_foot == "left" else _LEFT_LEG

    # Flat terrain only: balancing on one leg on rough ground is a different
    # (much harder) task. Inherits the walk robot model — nothing but the feet
    # should ever touch the floor here.
    cfg = make_microduck_velocity_env_cfg(play=play, rough=False)

    # ── Sensors: per-foot contact, so support and swing can be judged apart ──
    support_sensor = ContactSensorCfg(
        name="support_foot_ground_contact",
        primary=ContactMatch(
            mode="geom", pattern=rf"^{support_foot}_foot_collision$", entity="robot"
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found",),
        reduce="netforce",
        num_slots=1,
    )
    swing_sensor = ContactSensorCfg(
        name="swing_foot_ground_contact",
        primary=ContactMatch(
            mode="geom", pattern=rf"^{swing_foot}_foot_collision$", entity="robot"
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found",),
        reduce="netforce",
        num_slots=1,
    )
    cfg.scene.sensors = tuple(cfg.scene.sensors) + (support_sensor, swing_sensor)

    # ── Rewards: drop the walking stack ───────────────────────────────────────
    for name in _WALKING_REWARDS:
        cfg.rewards.pop(name, None)

    _phase = {
        "command_name": "twist",
        "descent_end": LIFT_END,
        "hold_end": HOLD_END,
        "rise_end": DOWN_END,
    }

    # ── Reward: the balance itself ────────────────────────────────────────────
    # THE task term. Standing on one leg is putting the whole-body CoM over that
    # foot; everything else in this file is scaffolding around this line. Uses
    # subtree_com (whole robot), NOT root_com_pos_w (trunk body only, 27% of the
    # mass) — see microduck_mdp.whole_body_com_w.
    cfg.rewards["com_over_support"] = RewardTermCfg(
        func=microduck_mdp.com_over_support_phased,
        weight=5.0,
        params={
            **_phase,
            "std": COM_SUPPORT_STD,
            "asset_cfg": SceneEntityCfg("robot", site_names=[f"{support_foot}_foot"]),
        },
    )

    # Hard state gate: the swing foot must actually leave the floor. A pose
    # target alone is satisfiable with the sole still scuffing the ground.
    cfg.rewards["swing_foot_air"] = RewardTermCfg(
        func=microduck_mdp.swing_foot_air_phased,
        weight=2.0,
        params={**_phase, "sensor_name": swing_sensor.name},
    )

    # Always-on anti-hop: the support foot must stay planted. Also kills the
    # obvious exploits — any hop or step loses this every step it is airborne.
    cfg.rewards["support_foot_grounded"] = RewardTermCfg(
        func=microduck_mdp.single_foot_grounded_reward,
        weight=2.0,
        params={"sensor_name": support_sensor.name},
    )

    # Rolling onto the inner edge of the support sole is THE failure mode here
    # (it is what happens when the CoM has not moved far enough). The sensor
    # gate zeroes this for the airborne foot, so only the stance sole is asked
    # to stay flat and the swing foot is free to hang at any angle.
    cfg.rewards["support_foot_flat"] = RewardTermCfg(
        func=microduck_mdp.feet_flat_penalty,
        weight=-2.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", site_names=["left_foot", "right_foot"]
            ),
            "sensor_name": "feet_ground_contact",
        },
    )

    # ── Rewards: the gesture ──────────────────────────────────────────────────
    # Phase-interpolated HOME ↔ FLAMINGO targets. Directive and symmetric: at
    # every instant they name the configuration wanted, and lowering the leg is
    # paid exactly like raising it, so there is no jackpot for getting up early.
    # They also restore a clean two-footed stand during the rest segment, which
    # the CoM term alone would not.
    #
    # The stds are LOOSE by design (see the measured one-sided authority in the
    # module docstring): these terms say what the trick looks like, the CoM term
    # says what it has to achieve, and the policy owns everything in between.
    def _subset(names):
        return {k: v for k, v in flamingo.items() if k in names}

    cfg.rewards["swing_pose"] = RewardTermCfg(
        func=microduck_mdp.phase_pose_track,
        weight=3.0,
        params={**_phase, "target_pose": _subset(swing_leg), "std": SWING_POSE_STD},
    )
    # L1 bootstrap: a constant gradient toward the target even where the
    # Gaussian has saturated near zero, i.e. before the lift exists at all.
    cfg.rewards["swing_pose_l1"] = RewardTermCfg(
        func=microduck_mdp.phase_pose_track_l1,
        weight=1.0,
        params={**_phase, "target_pose": _subset(swing_leg)},
    )
    cfg.rewards["support_pose"] = RewardTermCfg(
        func=microduck_mdp.phase_pose_track,
        weight=2.0,
        params={**_phase, "target_pose": _subset(support_leg), "std": SUPPORT_POSE_STD},
    )
    cfg.rewards["head_pose"] = RewardTermCfg(
        func=microduck_mdp.phase_pose_track,
        weight=1.0,
        params={
            **_phase,
            "target_pose": _subset(_HEAD_SAGITTAL),
            "std": HEAD_POSE_STD,
        },
    )

    # ── Rewards: regularisation ───────────────────────────────────────────────
    # upright: REVERTED to mjlab's base values (weight 1.0, std 0.447) from the
    # velocity recipe's deliberately strong 2.0 / 0.224. The measured keyframe
    # needs 19.3° of LATERAL trunk lean — that is not a defect, it is the only
    # way a robot with no ankle-roll DOF gets its CoM over one foot. At the
    # walking setting that lean costs ~1.8 of 2.0 per step, i.e. the reward
    # stack would be paying the policy to fall over. At 1.0 / 0.447 it costs
    # ~0.44, enough to keep the trunk from flopping without vetoing the task.
    cfg.rewards["upright"].weight = 1.0
    cfg.rewards["upright"].params["std"] = math.sqrt(0.2)

    # body_ang_vel / angular_momentum are motion-blockers, and active balancing
    # is exactly the corrective body motion they block (AGENTS: keep LOW for
    # dynamic tasks). Halved from the walking values rather than removed — some
    # damping still helps a hold.
    cfg.rewards["body_ang_vel"].weight = -0.02
    cfg.rewards["angular_momentum"].weight = -0.01

    # dof_pos_limits, restricted. The stock term fires in the last ~7.5% of a
    # joint's range and the velocity recipe applies it to everything — but this
    # task REQUIRES the support hip's roll and yaw right out near their stops
    # (that is where the 7.0 mm of margin lives, vs 2.8 mm at 90%). Leaving it
    # unrestricted would penalise the one thing that makes the task possible.
    # Every other joint keeps the guard.
    cfg.rewards["dof_pos_limits"].params["asset_cfg"] = SceneEntityCfg(
        "robot", joint_names=(r"^(?!passive_)(?!.*hip_(roll|yaw)).*",)
    )

    # Smoothness: stage-0 values, ramped by curriculum below. An attempt-tax
    # active while a hard skill is being explored makes "do nothing" win.
    cfg.rewards["action_rate_l2"].weight = -0.1
    cfg.rewards["joint_torques_l2"] = RewardTermCfg(
        func=microduck_mdp.joint_torques_l2, weight=-1e-3
    )

    # ── Command: the walking twist becomes a phase signal ─────────────────────
    _phase_fields = {
        f.name for f in dataclasses.fields(microduck_mdp.GroundPickPhaseCommandCfg)
    }
    _twist_kwargs = {
        k: deepcopy(v)
        for k, v in vars(cfg.commands["twist"]).items()
        if k in _phase_fields
    }
    _twist_kwargs.update(
        class_type=microduck_mdp.GroundPickPhaseCommand,
        period=ONE_LEG_PERIOD,
        # Random start phase per episode decorrelates envs. The runtime always
        # triggers at φ=0 from standing, which the rest segment covers.
        randomize_phase=True,
        rel_standing_envs=0.0,
        rel_heading_envs=0.0,
    )
    cfg.commands["twist"] = microduck_mdp.GroundPickPhaseCommandCfg(**_twist_kwargs)

    # ── Commands / obs: zero-pad the head + body pose slots ───────────────────
    # The head is a balance actuator here, so tracking a head command too would
    # fight the task. The slots stay in the obs (constant zero) to keep the
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
    # friction, mass-inertia / joint-friction / armature DR, encoder bias) is
    # inherited from the velocity env unchanged.
    if "push_robot" in cfg.events:
        cfg.events["push_robot"] = EventTermCfg(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(3.0, 6.0) if play else PUSH_INTERVAL_S,
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

    # CoM DR, capped at ±5 mm (see COM_DR_CAP). The inherited curricula ramp to
    # ±15 / ±10 mm, which exceeds the entire 7 mm static margin.
    for cur_name, cap in (("com_range", COM_DR_CAP), ("head_com_range", HEAD_COM_DR_CAP)):
        if cur_name in cfg.curriculum:
            stages = cfg.curriculum[cur_name].params["range_stages"]
            for stage in stages:
                stage["range"] = min(stage["range"], cap)

    # Smoothness AFTER discovery, and lighter than polite_bow's: a bow is a
    # quasi-static gesture, a one-leg hold is continuous active correction, and
    # action_rate taxes exactly those corrections.
    cfg.curriculum["action_rate_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "action_rate_l2",
            "weight_stages": [
                {"step": 0,                       "weight": -0.1},
                {"step": 400 * NUM_STEPS_PER_ENV, "weight": -0.3},
                {"step": 800 * NUM_STEPS_PER_ENV, "weight": -0.6},
            ],
        },
    )
    # Pushes ramp in only once the balance is consolidated — and stay small.
    cfg.curriculum["push_magnitude"] = CurriculumTermCfg(
        func=microduck_mdp.push_curriculum,
        params={
            "event_name": "push_robot",
            "push_stages": [
                {"step": 0,
                 "velocity_range": {"x": PUSH_RANGE_INITIAL, "y": PUSH_RANGE_INITIAL}},
                {"step": 800 * NUM_STEPS_PER_ENV,
                 "velocity_range": {"x": (-0.06, 0.06), "y": (-0.06, 0.06)}},
                {"step": 1400 * NUM_STEPS_PER_ENV,
                 "velocity_range": {"x": PUSH_RANGE_FINAL, "y": PUSH_RANGE_FINAL}},
            ],
        },
    )

    return cfg


# ── RL runner config ──────────────────────────────────────────────────────────

MicroduckOneLegRlCfg = RslRlOnPolicyRunnerCfg(
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
    experiment_name=f"one_leg_{SUPPORT_FOOT}",
    run_name=f"one_leg_{SUPPORT_FOOT}",
    save_interval=250,
    num_steps_per_env=NUM_STEPS_PER_ENV,
    max_iterations=20_000,
)
