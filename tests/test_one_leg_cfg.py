"""Cfg invariants for the one-leg-balance task.

CPU-only. These lock in the things the pre-training measurement established:
the phase window the runtime expects, the sign conventions, and — above all —
that the regularisers which would make a one-leg stand physically impossible
stay relaxed. See the module docstring of microduck_one_leg_env_cfg for the
measured numbers each of these guards.
"""

import math

import pytest

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_one_leg_env_cfg import (
    COM_DR_CAP,
    COM_SUPPORT_STD,
    DOWN_END,
    FLAMINGO_POSE,
    HEAD_COM_DR_CAP,
    HOLD_END,
    LIFT_END,
    ONE_LEG_PERIOD,
    SUPPORT_POSE_STD,
    SWING_POSE_STD,
    _FLAMINGO_LEFT_SUPPORT,
    _mirror_pose,
    make_microduck_one_leg_env_cfg,
)
from mjlab_microduck.robot.microduck_constants import HOME_FRAME


@pytest.fixture(scope="module")
def cfg():
    return make_microduck_one_leg_env_cfg()


# ── Command / phase ───────────────────────────────────────────────────────────

def test_command_is_a_phase_signal(cfg):
    cmd = cfg.commands["twist"]
    assert isinstance(cmd, microduck_mdp.GroundPickPhaseCommandCfg)
    assert cmd.class_type is microduck_mdp.GroundPickPhaseCommand
    # Must match --ground-pick-period at deploy and duration_s in the manifest.
    assert cmd.period == ONE_LEG_PERIOD == 6.0


def test_lower_finishes_before_the_runtime_hands_control_back():
    """scripts/infer_policy.py ends the ground-pick slot at phase >= 0.7.

    A profile still lowering the leg at 0.7 is cut off mid-motion and the
    walking policy inherits a raised knee.
    """
    assert 0.0 < LIFT_END < HOLD_END < DOWN_END < 0.7


def test_hold_is_long_enough_to_read_as_a_hold():
    """The point of the task is the hold, not the lift."""
    assert (HOLD_END - LIFT_END) * ONE_LEG_PERIOD >= 1.5


# ── The keyframe ──────────────────────────────────────────────────────────────

def _home_of(joint_name: str) -> float:
    """HOME_FRAME keys are regex patterns (and some are shared across sides,
    e.g. r'.*hip_yaw.*'), so resolve by matching, first pattern wins — the
    same order mjlab applies them in."""
    import re
    for pattern, value in HOME_FRAME.joint_pos.items():
        if re.match(pattern, joint_name):
            return value
    raise AssertionError(f"no HOME entry matches {joint_name}")


def test_keyframe_only_contains_joints_that_move():
    """polite_bow's lesson: a joint pinned at its HOME value scores a perfect
    1.0 every step and dilutes the Gaussian averaged over the tracked set."""
    for name, val in _FLAMINGO_LEFT_SUPPORT.items():
        home = _home_of(name)
        assert abs(val - home) > 0.1, f"{name} barely moves from HOME ({val} vs {home})"


def test_keyframe_stays_inside_the_mechanical_limits():
    """hip_roll is +-22deg and hip_yaw asymmetric -- both CAD-derived limits.

    The keyframe deliberately stops at ~90% of range so the policy keeps
    overshoot headroom in the one direction that can save a topple.
    """
    limits = {
        "hip_roll": (-0.3840, 0.3840),
        "left_hip_yaw": (-0.4363, 0.5236),
        "right_hip_yaw": (-0.5236, 0.4363),
        "neck_pitch": (-1.5708, 1.0472),
    }
    for name, val in _FLAMINGO_LEFT_SUPPORT.items():
        key = name if name in limits else next(
            (k for k in limits if k in name), None
        )
        if key is None:
            continue
        lo, hi = limits[key]
        assert lo < val < hi, f"{name}={val} outside [{lo}, {hi}]"
        # ...and not jammed against the stop.
        assert min(val - lo, hi - val) > 0.02, f"{name} parks on its limit"


def test_head_yaw_and_roll_are_left_unpriced():
    """They are worth ~3 mm of the 7 mm static margin as a counterweight.

    No keyframe entry and no HOME term anywhere: the policy has to be free to
    discover the counterweight.
    """
    assert "head_yaw" not in FLAMINGO_POSE
    assert "head_roll" not in FLAMINGO_POSE


def test_mirror_pose_is_an_involution():
    once = _mirror_pose(_FLAMINGO_LEFT_SUPPORT)
    twice = _mirror_pose(once)
    assert set(twice) == set(_FLAMINGO_LEFT_SUPPORT)
    for k, v in _FLAMINGO_LEFT_SUPPORT.items():
        assert twice[k] == pytest.approx(v)
    # The support leg really did swap sides.
    assert once["right_hip_roll"] == pytest.approx(-_FLAMINGO_LEFT_SUPPORT["left_hip_roll"])
    # Sagittal head joints are not mirrored.
    assert once["neck_pitch"] == _FLAMINGO_LEFT_SUPPORT["neck_pitch"]


def test_right_support_variant_builds_and_mirrors(cfg):
    right = make_microduck_one_leg_env_cfg(support_foot="right")
    assert right.rewards["com_over_support"].params["asset_cfg"].site_names == ["right_foot"]
    assert cfg.rewards["com_over_support"].params["asset_cfg"].site_names == ["left_foot"]
    # The swing pose targets the OTHER leg in each case.
    assert all(k.startswith("right_") for k in cfg.rewards["swing_pose"].params["target_pose"])
    assert all(k.startswith("left_") for k in right.rewards["swing_pose"].params["target_pose"])


# ── Reward stack ──────────────────────────────────────────────────────────────

def test_com_over_support_is_the_main_term(cfg):
    """Balancing on one leg IS putting the CoM over that foot."""
    r = cfg.rewards["com_over_support"]
    assert r.func is microduck_mdp.com_over_support_phased
    assert r.weight > 0
    assert r.params["std"] == COM_SUPPORT_STD
    # It must outweigh any single pose term, or the gesture becomes the goal.
    for pose_term in ("swing_pose", "support_pose", "head_pose"):
        assert r.weight > cfg.rewards[pose_term].weight


def test_whole_body_com_not_trunk_com():
    """root_com_pos_w is the TRUNK body alone (0.199 of 0.737 kg).

    The head is 38% of the mass on a long lever; using the trunk's own CoM
    would measure the wrong thing entirely.
    """
    import inspect
    src = inspect.getsource(microduck_mdp.com_over_support_phased)
    assert "whole_body_com_w" in src
    assert "root_com_pos_w" not in src
    assert "subtree_com" in inspect.getsource(microduck_mdp.whole_body_com_w)


def test_foot_contact_terms(cfg):
    r = cfg.rewards
    # Swing foot must actually leave the floor (hard state gate).
    assert r["swing_foot_air"].func is microduck_mdp.swing_foot_air_phased
    assert r["swing_foot_air"].weight > 0
    assert r["swing_foot_air"].params["sensor_name"] == "swing_foot_ground_contact"
    # Support foot planted, always on (anti-hop).
    assert r["support_foot_grounded"].weight > 0
    assert r["support_foot_grounded"].params["sensor_name"] == "support_foot_ground_contact"
    # Rolling onto the sole's inner edge is the failure mode; the sensor gate
    # frees the airborne foot so only the stance sole is asked to stay flat.
    assert r["support_foot_flat"].weight < 0
    assert r["support_foot_flat"].params["sensor_name"] == "feet_ground_contact"


def test_per_foot_sensors_exist(cfg):
    names = {s.name for s in cfg.scene.sensors}
    assert {"support_foot_ground_contact", "swing_foot_ground_contact"} <= names


def test_pose_tracking_is_deliberately_loose(cfg):
    """The measured authority at the balance optimum is one-sided: 41.7 mm back
    toward the midline, 1.1 mm further over the foot. The swing leg and head are
    the only correction levers left, so tight pose tracking would price away the
    balancing itself.
    """
    assert cfg.rewards["swing_pose"].params["std"] == SWING_POSE_STD >= 0.3
    assert cfg.rewards["support_pose"].params["std"] == SUPPORT_POSE_STD >= 0.45
    assert cfg.rewards["head_pose"].params["std"] >= 0.45
    # Support tracking must be looser than swing: it is a suggestion, the CoM
    # term is the objective.
    assert cfg.rewards["support_pose"].params["std"] > cfg.rewards["swing_pose"].params["std"]
    # L1 bootstrap on the swing leg, self-negating so the weight is POSITIVE.
    assert cfg.rewards["swing_pose_l1"].func is microduck_mdp.phase_pose_track_l1
    assert cfg.rewards["swing_pose_l1"].weight > 0


# ── The regularisers that would make the task impossible ─────────────────────

def test_upright_is_relaxed_for_the_required_lateral_lean(cfg):
    """The keyframe needs 19.3 deg of lateral trunk lean -- forced, since there
    is no ankle-roll DOF. At the walking setting (2.0 / std 0.224) that lean
    costs ~1.8 of 2.0 per step, i.e. the stack would pay the policy to fall.
    """
    up = cfg.rewards["upright"]
    lean = math.sin(math.radians(19.3))
    cost_fraction = 1.0 - math.exp(-((lean / up.params["std"]) ** 2))
    assert up.weight * cost_fraction < 0.6, (
        f"upright taxes the required lean by {up.weight * cost_fraction:.2f}/step"
    )


def test_motion_blockers_stay_low(cfg):
    """body_ang_vel / angular_momentum penalise exactly the corrective body
    motion that active balancing requires (AGENTS: keep LOW for dynamic tasks).
    """
    assert -0.05 < cfg.rewards["body_ang_vel"].weight <= 0.0
    assert -0.02 < cfg.rewards["angular_momentum"].weight <= 0.0


def test_dof_pos_limits_spares_the_hips_that_do_the_balancing(cfg):
    """The 7 mm of static margin lives out near the hip roll/yaw stops (2.8 mm
    at 90% of range). The stock term fires in the last ~7.5% of range, so left
    unrestricted it would penalise the one thing that makes the task possible.
    """
    import re
    patterns = cfg.rewards["dof_pos_limits"].params["asset_cfg"].joint_names
    assert patterns, "dof_pos_limits must be restricted, not left at the default"
    rx = [re.compile(p) for p in patterns]
    for spared in ("left_hip_roll", "right_hip_roll", "left_hip_yaw", "right_hip_yaw"):
        assert not any(r.match(spared) for r in rx), f"{spared} still guarded"
    for guarded in ("left_knee", "right_ankle", "neck_pitch", "left_hip_pitch"):
        assert any(r.match(guarded) for r in rx), f"{guarded} lost its guard"
    # passive_* joints stay excluded (repo-wide convention).
    assert not any(r.match("passive_left_knee_backlash") for r in rx)


def test_com_dr_is_capped_below_the_static_margin(cfg):
    """The best static margin is 7 mm. The walking recipe ramps trunk CoM DR to
    +-15 mm and head CoM to +-10 mm -- either alone exceeds the whole margin, so
    those envs would be kinematically unable to balance.
    """
    assert COM_DR_CAP <= 0.007 and HEAD_COM_DR_CAP <= 0.007
    for cur_name, cap in (("com_range", COM_DR_CAP), ("head_com_range", HEAD_COM_DR_CAP)):
        stages = cfg.curriculum[cur_name].params["range_stages"]
        assert stages, cur_name
        assert max(s["range"] for s in stages) <= cap, cur_name


def test_pushes_start_tiny_and_ramp(cfg):
    """+-0.3 m/s (the walking value) topples a one-leg stand outright."""
    stages = cfg.curriculum["push_magnitude"].params["push_stages"]
    assert stages[0]["step"] == 0
    first = abs(stages[0]["velocity_range"]["x"][1])
    last = abs(stages[-1]["velocity_range"]["x"][1])
    assert first < last <= 0.15
    # And the ramp must wait for the balance to exist.
    assert stages[1]["step"] >= 500 * 24


def test_smoothness_is_introduced_after_the_skill(cfg):
    stages = cfg.curriculum["action_rate_weight"].params["weight_stages"]
    assert stages[0]["step"] == 0
    weights = [s["weight"] for s in stages]
    assert weights == sorted(weights, reverse=True)
    # Lighter than polite_bow's -2.0: a bow is quasi-static, a one-leg hold is
    # continuous active correction and action_rate taxes those corrections.
    assert weights[-1] >= -1.0


# ── Inherited stack / obs contract ───────────────────────────────────────────

def test_walking_objective_is_gone(cfg):
    for gone in (
        "track_linear_velocity", "track_angular_velocity", "air_time",
        "foot_clearance", "foot_swing_height", "foot_slip", "pose",
        "head_pose_tracking", "head_pose_bias", "body_pose_tracking",
    ):
        assert gone not in cfg.rewards, gone
    for gone in ("standing_envs", "head_pose_range", "body_pose_range",
                 "head_pose_bias_weight"):
        assert gone not in cfg.curriculum, gone
    assert "head_pose" not in cfg.commands
    assert "body_pose" not in cfg.commands


def test_obs_keeps_the_unified_61d_layout(cfg):
    """The 10 trailing command slots are zero-padded, never deleted."""
    for group in ("actor", "critic"):
        terms = cfg.observations[group].terms
        assert terms["head_command"].func is microduck_mdp.zero_command_padding
        assert terms["head_command"].params["dim"] == 4
        assert terms["body_command"].func is microduck_mdp.zero_command_padding
        assert terms["body_command"].params["dim"] == 6
        names = list(terms.keys())
        assert names.index("head_command") < names.index("body_command")


def test_nan_guard_and_bam_events_inherited(cfg):
    """Inherited from the velocity recipe — a standalone build would drop these."""
    assert "nan_state" in cfg.terminations
    assert "expand_bam_friction_fields" in cfg.events
    assert "reset_action_history" in cfg.events
    assert "randomize_joint_friction" in cfg.events  # BAM friction_scale path
    assert cfg.observations["actor"].terms["joint_pos"].params["biased"] is True


def test_every_penalty_has_a_sign_that_costs(cfg):
    """AGENTS' infallible check, at cfg time: mjlab cost functions (>=0) need a
    negative weight; microduck's self-negating *_penalty / *_l1 need a positive
    one. A negative weight on a self-negating penalty rewards the violation.
    """
    self_negating = {"swing_pose_l1"}
    mjlab_costs = {
        "body_ang_vel", "angular_momentum", "dof_pos_limits", "action_rate_l2",
        "self_collisions", "support_foot_flat", "joint_torques_l2",
    }
    for name in self_negating:
        assert cfg.rewards[name].weight > 0, name
    for name in mjlab_costs:
        assert cfg.rewards[name].weight < 0, name


def test_symmetry_stays_off():
    """Standing on one leg is inherently asymmetric."""
    from mjlab_microduck.tasks.microduck_one_leg_env_cfg import (
        ENABLE_SYMMETRY,
        MicroduckOneLegRlCfg,
    )
    assert ENABLE_SYMMETRY is False
    assert MicroduckOneLegRlCfg.algorithm.symmetry_cfg is None


def test_play_variant_builds():
    play = make_microduck_one_leg_env_cfg(play=True)
    assert "com_over_support" in play.rewards
    assert play.events["push_robot"].interval_range_s == (3.0, 6.0)


# ── Numeric checks of the phase trajectory and the new mdp functions ─────────
# Structure is locked above; these check the motion the reward actually asks
# for, using the same fake-env harness as test_ground_pick_pose.

import torch  # noqa: E402
from mjlab.managers.scene_entity_config import SceneEntityCfg  # noqa: E402

from test_ground_pick_pose import _FakeEnv  # noqa: E402  (pytest puts tests/ on sys.path)

_SWING = ["right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee"]
_SWING_HOME = torch.tensor([[0.0, 0.0873, 0.4579, 0.0049]])
_SWING_FLAMINGO = torch.tensor([[_FLAMINGO_LEFT_SUPPORT[n] for n in _SWING]])
_SWING_TARGET = {n: _FLAMINGO_LEFT_SUPPORT[n] for n in _SWING}


def _swing_env(cur, phase):
    return _FakeEnv(_SWING, cur.clone(), _SWING_HOME.clone(), phase)


def _track(cur, phase):
    return microduck_mdp.phase_pose_track(
        _swing_env(cur, phase),
        target_pose=_SWING_TARGET,
        std=SWING_POSE_STD,
        descent_end=LIFT_END,
        hold_end=HOLD_END,
        rise_end=DOWN_END,
        asset_cfg=SceneEntityCfg("robot"),
    )


@pytest.mark.parametrize(
    "phase,expected",
    [
        (0.0, _SWING_HOME),                      # trigger: standing
        (LIFT_END, _SWING_FLAMINGO),             # knee up
        ((LIFT_END + HOLD_END) / 2, _SWING_FLAMINGO),  # still held
        (DOWN_END, _SWING_HOME),                 # back down
        (0.9, _SWING_HOME),                      # resting
    ],
)
def test_swing_trajectory_lifts_and_returns(phase, expected):
    assert torch.allclose(_track(expected, phase), torch.tensor([1.0]), atol=1e-6)


def test_lifting_early_is_not_a_jackpot():
    """Being already at the flamingo halfway up the ramp scores WORSE than
    tracking the ramp — the anti-jackpot property the whole profile rests on."""
    half_phase = LIFT_END / 2
    on_ramp = (_SWING_HOME + _SWING_FLAMINGO) / 2
    assert torch.allclose(_track(on_ramp, half_phase), torch.tensor([1.0]), atol=1e-6)
    assert _track(_SWING_FLAMINGO, half_phase).item() < 1.0


def test_standing_still_never_scores_a_full_lift():
    """A policy that just stands there must not collect the swing reward."""
    assert _track(_SWING_HOME, HOLD_END - 0.01).item() < 0.75


def test_swing_gradient_is_worth_chasing():
    """Episode_Reward/swing_pose is mean(func) * weight over a full cycle, so
    these are directly the wandb numbers to expect. The loose std that protects
    the balance authority must not flatten the signal to nothing.
    """
    from mjlab_microduck.tasks.mdp import phase_pose_blend

    weight = 3.0
    phases = torch.linspace(0, 1, 201)[:-1]

    def mean_over_cycle(pose_at):
        return sum(_track(pose_at(float(p)), float(p)).item() for p in phases) / len(phases)

    def on_the_ramp(ph):
        b = phase_pose_blend(torch.tensor([ph]), LIFT_END, HOLD_END, DOWN_END)
        return _SWING_HOME + b.unsqueeze(-1) * (_SWING_FLAMINGO - _SWING_HOME)

    stand_still = mean_over_cycle(lambda _: _SWING_HOME) * weight
    perfect = mean_over_cycle(on_the_ramp) * weight
    assert perfect == pytest.approx(3.0, abs=0.01)
    assert perfect - stand_still > 0.8, (
        f"only {perfect - stand_still:.2f} between lifting and standing still"
    )


# ── swing_foot_air_phased / com_over_support_phased ──────────────────────────

class _FakeSensor:
    def __init__(self, found):
        self.data = type("D", (), {"found": found})()


class _FakeScene(dict):
    def __init__(self, sensors, entities=None):
        super().__init__(entities or {})
        self.sensors = sensors


class _AirEnv:
    def __init__(self, contact, phase):
        import math
        self.device = "cpu"
        self.num_envs = 1
        self.scene = _FakeScene(
            {"swing": _FakeSensor(torch.tensor([[float(contact)]]))}
        )
        ang = 2 * math.pi * phase
        self.command_manager = type(
            "C", (), {"get_command": lambda _s, _n: torch.tensor([[math.cos(ang), math.sin(ang), 0.0]])}
        )()


def _air(contact, phase):
    return microduck_mdp.swing_foot_air_phased(
        _AirEnv(contact, phase), sensor_name="swing",
        command_name="twist", descent_end=LIFT_END, hold_end=HOLD_END, rise_end=DOWN_END,
    ).item()


def test_swing_foot_air_pays_only_for_an_actually_lifted_foot():
    mid_hold = (LIFT_END + HOLD_END) / 2
    assert _air(contact=0, phase=mid_hold) == pytest.approx(1.0)   # up, during hold
    assert _air(contact=1, phase=mid_hold) == pytest.approx(0.0)   # scuffing: nothing
    # During rest the foot belongs on the floor, and keeping it up pays nothing.
    assert _air(contact=1, phase=0.9) == pytest.approx(0.0)
    assert _air(contact=0, phase=0.9) == pytest.approx(0.0)
    # Lifting ahead of the ramp is paid by the ramp, not in full.
    assert 0.0 < _air(contact=0, phase=LIFT_END / 2) < 1.0


def test_swing_foot_air_is_safe_without_the_sensor():
    """Missing sensor must degrade to 0, not raise (mirrors the other
    sensor-backed terms — a raise here would kill training mid-run)."""
    env = _AirEnv(contact=0, phase=0.3)
    assert microduck_mdp.swing_foot_air_phased(env, sensor_name="absent").item() == 0.0


# ── Self-collision: the check the original keyframe solve was missing ────────
# The first shipped FLAMINGO keyframe drove the swing shank 8.07 mm into the
# trunk battery holder. The pose solve had constrained CoM margin and swing-foot
# clearance but never self-contact, and nothing here caught it — the policy just
# quietly learned to press its leg against its own body. This test is the guard.
#
# Oracle note: mj_geomDistance returns exactly 0.0 for these mesh-mesh pairs when
# they are apart, so it CANNOT distinguish "far" from "touching" and must not be
# used. Inflating geom_margin instead makes MuJoCo emit a contact whenever the
# surfaces are within the probe distance, and contact.dist is then the true
# signed separation — so "no contact at probe P" proves clearance > P.

_SELF_COLLISION_PROBE = 0.020  # 20 mm


def _self_collision_model():
    import mujoco
    from mjlab_microduck.robot.microduck_constants import MICRODUCK_WALK_XML
    m = mujoco.MjModel.from_xml_path(str(MICRODUCK_WALK_XML))
    self_geoms = [i for i in range(m.ngeom) if m.geom_contype[i] == 2]
    assert len(self_geoms) >= 2, "walk model lost its self-collision geoms"
    for g in self_geoms:
        m.geom_margin[g] = _SELF_COLLISION_PROBE
    return m, mujoco.MjData(m), set(self_geoms)


def _clearance_along_ramp(pose: dict, samples: int = 81):
    """Worst self-clearance over the commanded HOME -> pose ramp.

    The reward tracks the INTERPOLATED target, so an endpoint-only check would
    still let the policy be commanded through an interpenetration on the way in.
    """
    import mujoco
    m, d, self_geoms = _self_collision_model()
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(m.njnt)]
    adr = {n: m.jnt_qposadr[i] for i, n in enumerate(names) if n}
    home = {n: _home_of(n) for n in adr if n and not n.startswith("passive_")
            and n != "trunk_base_freejoint"}

    worst, worst_t = _SELF_COLLISION_PROBE, 0.0
    for k in range(samples):
        t = k / (samples - 1)
        d.qpos[:] = 0.0
        d.qpos[3] = 1.0
        for n, h in home.items():
            d.qpos[adr[n]] = h + t * (pose.get(n, h) - h)
        mujoco.mj_forward(m, d)
        hits = [c.dist for c in d.contact[:d.ncon]
                if c.geom1 in self_geoms and c.geom2 in self_geoms]
        if hits and min(hits) < worst:
            worst, worst_t = min(hits), t
    return worst, worst_t


def test_oracle_detects_a_known_interpenetrating_pose():
    """Guard the guard: the original keyframe must still read as colliding."""
    bad = dict(_FLAMINGO_LEFT_SUPPORT)
    bad["right_hip_pitch"] = -1.1335   # the pose that shipped, -64.9 deg
    bad["right_hip_yaw"] = 0.3927
    worst, _ = _clearance_along_ramp(bad, samples=41)
    assert worst < 0, (
        "the self-collision oracle no longer detects the pose that motivated it "
        f"(got {worst * 1000:.2f} mm) — the check has silently stopped working"
    )


def test_flamingo_keyframe_is_self_collision_free():
    """The commanded pose, and every point on the ramp to it, must not drive the
    robot into its own battery holder."""
    worst, t = _clearance_along_ramp(_FLAMINGO_LEFT_SUPPORT)
    assert worst > 0, (
        f"FLAMINGO keyframe commands self-interpenetration of "
        f"{-worst * 1000:.2f} mm at blend {t:.2f}"
    )
