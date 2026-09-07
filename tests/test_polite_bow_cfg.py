"""Cfg invariants for the polite-bow task.

CPU-only: these lock in the sign conventions, the phase window the runtime
expects, and the fact that the walking objective is really gone.
"""

import pytest

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_polite_bow_env_cfg import (
    BOW_POSE,
    BOW_PERIOD,
    DESCENT_END,
    HOLD_END,
    RISE_END,
    make_microduck_polite_bow_env_cfg,
)
from mjlab_microduck.robot.microduck_constants import HOME_FRAME


@pytest.fixture(scope="module")
def cfg():
    return make_microduck_polite_bow_env_cfg()


def test_command_is_a_phase_signal(cfg):
    cmd = cfg.commands["twist"]
    assert isinstance(cmd, microduck_mdp.GroundPickPhaseCommandCfg)
    assert cmd.class_type is microduck_mdp.GroundPickPhaseCommand
    # Must match --ground-pick-period at deploy and duration_s in the manifest.
    assert cmd.period == BOW_PERIOD == 4.0


def test_rise_finishes_before_the_runtime_hands_control_back():
    """scripts/infer_policy.py ends the ground-pick slot at phase >= 0.7.

    A profile whose rise is still running at 0.7 gets cut off mid-motion and the
    walking policy inherits a bowed head (ground_pick carries a warning about
    exactly this).
    """
    assert 0.0 < DESCENT_END < HOLD_END < RISE_END < 0.7


def test_bow_pose_directions_match_the_measured_kinematics():
    """neck_pitch DOWN and head_pitch UP are what tip the head forward/down."""
    home_neck = HOME_FRAME.joint_pos[r".*neck_pitch.*"]
    home_head = HOME_FRAME.joint_pos[r".*head_pitch.*"]
    assert BOW_POSE["neck_pitch"] < home_neck  # neck swings the head forward+down
    assert BOW_POSE["head_pitch"] > home_head  # beak tips down
    # ONLY moving joints belong in the tracked target: a joint pinned at its HOME
    # value scores a perfect 1.0 every step and dilutes the Gaussian mean.
    # head_yaw / head_roll are held by the separate `bow_sagittal` term.
    assert set(BOW_POSE) == {"neck_pitch", "head_pitch"}
    # Stay inside the mechanical range (neck_pitch is the tight one: [-1.57, 1.05]).
    assert -1.5 < BOW_POSE["neck_pitch"] < 1.0
    assert -1.5 < BOW_POSE["head_pitch"] < 1.5


def test_bow_reward_stack(cfg):
    r = cfg.rewards
    # Phase-interpolated head pose + its L1 bootstrap, both positive weights.
    assert r["bow_pose"].func is microduck_mdp.phase_pose_track
    assert r["bow_pose"].weight > 0
    assert r["bow_pose_l1"].func is microduck_mdp.phase_pose_track_l1
    assert r["bow_pose_l1"].weight > 0  # the func self-negates
    assert r["bow_pose"].params["target_pose"] is BOW_POSE
    # Legs hold the standing stance; the bow stays sagittal; feet planted+flat.
    assert r["leg_stance"].weight > 0
    assert r["bow_sagittal"].weight > 0
    assert r["bow_sagittal"].params["joint_indices"] == [7, 8]  # head_yaw, head_roll
    assert r["feet_grounded"].weight > 0
    assert r["feet_flat"].weight < 0
    # Trunk stays vertical: this is a HEAD bow.
    assert r["upright"].weight > 0


def test_walking_objective_is_gone(cfg):
    for gone in (
        "track_linear_velocity",
        "track_angular_velocity",
        "air_time",
        "foot_clearance",
        "foot_swing_height",
        "foot_slip",
        "pose",
        "head_pose_tracking",
        "head_pose_bias",
        "body_pose_tracking",
    ):
        assert gone not in cfg.rewards, gone
    for gone in ("standing_envs", "head_pose_range", "body_pose_range",
                 "head_pose_bias_weight"):
        assert gone not in cfg.curriculum, gone
    # The head/body command terms are dropped, not merely unweighted.
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
        # Order matters: [twist(3), head_pose(4), body_pose(6)].
        names = list(terms.keys())
        assert names.index("head_command") < names.index("body_command")


def test_smoothness_is_introduced_after_the_skill(cfg):
    """An attempt-tax active during discovery makes 'hold still' win."""
    neck_stages = cfg.curriculum["neck_action_rate_weight"].params["weight_stages"]
    assert neck_stages[0]["step"] == 0 and neck_stages[0]["weight"] == 0.0
    assert neck_stages[-1]["weight"] < 0
    action_stages = cfg.curriculum["action_rate_weight"].params["weight_stages"]
    # Monotonically heavier, and heavier than walking's final -1.0 (slow careful
    # motion wants more damping than a gait).
    weights = [s["weight"] for s in action_stages]
    assert weights == sorted(weights, reverse=True)
    assert weights[-1] <= -2.0


def test_pushes_ramp_in_after_the_bow_exists(cfg):
    stages = cfg.curriculum["push_magnitude"].params["push_stages"]
    first = stages[0]["velocity_range"]["x"]
    last = stages[-1]["velocity_range"]["x"]
    assert stages[0]["step"] == 0
    assert abs(first[1]) < abs(last[1])


def test_nan_guard_and_bam_events_inherited(cfg):
    """Inherited from the velocity recipe — a standalone build would drop these."""
    assert "nan_state" in cfg.terminations
    assert "expand_bam_friction_fields" in cfg.events
    assert "reset_action_history" in cfg.events
    assert "randomize_joint_friction" in cfg.events  # BAM friction_scale path
    assert cfg.observations["actor"].terms["joint_pos"].params["biased"] is True


def test_play_variant_builds():
    play = make_microduck_polite_bow_env_cfg(play=True)
    assert "bow_pose" in play.rewards
    # Play pushes are spaced out, not the velocity env's 0.5-1.0 s stress test.
    assert play.events["push_robot"].interval_range_s == (2.0, 4.0)


# ── Numeric check of the phase trajectory ─────────────────────────────────────
# The cfg tests above lock in structure; this one checks the motion the reward
# actually asks for, using the same fake-env harness as test_ground_pick_pose.

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg

from test_ground_pick_pose import _FakeEnv  # noqa: E402  (pytest puts tests/ on sys.path)

_HEAD_NAMES = ["neck_pitch", "head_pitch"]
_HOME = torch.tensor([[0.3491, 0.3491]])
_BOW = torch.tensor([[BOW_POSE[n] for n in _HEAD_NAMES]])


def _bow_env(cur, phase):
    return _FakeEnv(_HEAD_NAMES, cur.clone(), _HOME.clone(), phase)


@pytest.mark.parametrize(
    "phase,expected",
    [
        (0.0,                        _HOME),  # trigger: standing
        (DESCENT_END,                _BOW),   # bottom of the bow
        ((DESCENT_END + HOLD_END) / 2, _BOW), # still held
        (RISE_END,                   _HOME),  # back up
        (0.9,                        _HOME),  # resting
    ],
)
def test_target_trajectory_reaches_bow_and_returns(phase, expected):
    """Tracking is perfect exactly when the head sits on the intended target."""
    r = microduck_mdp.phase_pose_track(
        _bow_env(expected, phase),
        target_pose=BOW_POSE,
        std=0.15,
        descent_end=DESCENT_END,
        hold_end=HOLD_END,
        rise_end=RISE_END,
        asset_cfg=SceneEntityCfg("robot"),
    )
    assert torch.allclose(r, torch.tensor([1.0]), atol=1e-6), r


def test_holding_still_never_scores_a_full_bow():
    """A policy that just stands there must not collect the bow reward."""
    r = microduck_mdp.phase_pose_track(
        _bow_env(_HOME, DESCENT_END),
        target_pose=BOW_POSE,
        std=0.15,
        descent_end=DESCENT_END,
        hold_end=HOLD_END,
        rise_end=RISE_END,
        asset_cfg=SceneEntityCfg("robot"),
    )
    assert r.item() < 0.6, r


def test_descent_is_gradual_not_a_step():
    """Halfway down the target is halfway to the bow — no jackpot for arriving early."""
    half = (_HOME + _BOW) / 2
    r = microduck_mdp.phase_pose_track(
        _bow_env(half, DESCENT_END / 2),
        target_pose=BOW_POSE,
        std=0.15,
        descent_end=DESCENT_END,
        hold_end=HOLD_END,
        rise_end=RISE_END,
        asset_cfg=SceneEntityCfg("robot"),
    )
    assert torch.allclose(r, torch.tensor([1.0]), atol=1e-6), r
    # ...and being ALREADY at the bow that early scores worse than tracking it.
    early = microduck_mdp.phase_pose_track(
        _bow_env(_BOW, DESCENT_END / 2),
        target_pose=BOW_POSE,
        std=0.15,
        descent_end=DESCENT_END,
        hold_end=HOLD_END,
        rise_end=RISE_END,
        asset_cfg=SceneEntityCfg("robot"),
    )
    assert early.item() < 1.0


def test_doing_nothing_scores_well_below_a_real_bow():
    """The tracked target must actually discriminate bowing from standing still.

    Episode_Reward/<term> equals mean(func) * weight for a full-length episode
    (reward_manager: episode sum / max_episode_length_s, each step scaled by dt),
    so these numbers are directly the wandb curve to expect.

    Regression guard: head_yaw/head_roll once sat in BOW_POSE at their HOME value.
    phase_pose_track averages its Gaussian over the joints it is given, so those
    two scored 1.0 every step regardless of behaviour and halved the gradient —
    standing still earned 4.55 of a possible 6.00 (76%). With only the moving
    joints tracked the floor drops to ~3.09 and the signal doubles.
    """
    from mjlab_microduck.tasks.mdp import phase_pose_blend

    weight = 6.0
    phases = torch.linspace(0, 1, 201)[:-1]

    def mean_over_cycle(pose_at):
        total = 0.0
        for ph in phases:
            r = microduck_mdp.phase_pose_track(
                _bow_env(pose_at(float(ph)), float(ph)),
                target_pose=BOW_POSE, std=0.15,
                descent_end=DESCENT_END, hold_end=HOLD_END, rise_end=RISE_END,
                asset_cfg=SceneEntityCfg("robot"),
            )
            total += r.item()
        return total / len(phases)

    def on_the_ramp(ph):
        blend = phase_pose_blend(torch.tensor([ph]), DESCENT_END, HOLD_END, RISE_END)
        return _HOME + blend.unsqueeze(-1) * (_BOW - _HOME)

    stand_still = mean_over_cycle(lambda ph: _HOME) * weight
    perfect = mean_over_cycle(on_the_ramp) * weight

    assert perfect == pytest.approx(6.0, abs=0.01)
    assert stand_still < 3.5, f"do-nothing floor too high ({stand_still:.2f}/6.00)"
    assert perfect - stand_still > 2.5, "not enough gradient between bowing and standing"
