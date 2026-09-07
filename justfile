# Task runner for mjlab-microduck. `just` with no arguments lists recipes.

policies_repo := "pollen-robotics/microduck-policies"

# HF namespace the Jobs bill to; the src tarball, checkpoint repo and uv cache
# bucket all live here. wandb entity is where --wandb-run-path resolves runs.
hf_namespace := "DavidMakesRobots"
wandb_entity := "dweis7-davidmakesrobots"

[private]
default:
    @just --list

# Print the local dir of the official policy set, downloading it on first use.
[private]
policies:
    @uv run python -c "from huggingface_hub import snapshot_download; print(snapshot_download('{{policies_repo}}'))"

# Sim with every walking-robot policy loaded (walk, stand, sit, pick, kick, roulade).
run-full-sim:
    #!/usr/bin/env bash
    set -euo pipefail
    D=$({{just_executable()}} policies)
    # macOS forces mujoco.viewer.launch_passive to run under `mjpython`.
    PY=$([ "$(uname -s)" = "Darwin" ] && echo mjpython || echo python)
    # Rollers are omitted on purpose: they need --roller, which loads a different
    # robot XML, and infer_policy.py refuses to mix that with kick/roulade.
    uv run "$PY" scripts/infer_policy.py \
        --new-cmd-obs \
        --walking     "$D/alpha_walking.onnx" \
        --standing    "$D/alpha_stand.onnx" \
        --sitstand    "$D/alpha_sitstand.onnx" \
        --ground-pick "$D/alpha_ground_pick.onnx" \
        --kick-left   "$D/ball_kick_left.onnx" \
        --kick-right  "$D/ball_kick_right.onnx" \
        --roulade     "$D/roulade.onnx"

# Sim on the roller-skate model (passive wheels under the feet). Own robot XML.
run-roller-sim:
    #!/usr/bin/env bash
    set -euo pipefail
    D=$({{just_executable()}} policies)
    PY=$([ "$(uname -s)" = "Darwin" ] && echo mjpython || echo python)
    # roller_crouch rides the ground-pick slot: both use the manifest's `phase`
    # encoding on twist.vx/vy, so G triggers the crouch, with the manifest's 5.0 s
    # period. Verified to load and balance; the crouch itself is not regression-tested.
    uv run "$PY" scripts/infer_policy.py \
        --new-cmd-obs \
        --roller \
        --walking           "$D/roller.onnx" \
        --ground-pick       "$D/roller_crouch.onnx" \
        --ground-pick-period 5.0

# `--gpu-ids None` is what puts mjlab in CPU mode: the default `[0]` dies in
# select_gpus on a machine without CUDA. Costs nothing and needs no wandb
# account. Run it after ANY cfg change and before paying for a GPU — it catches
# ~95% of config errors: obs shape, every reward term computing, penalty signs,
# NaN-freedom, ONNX export.

# Smoke-test the polite bow locally on CPU (64 envs, 5 iters, ~15 s).
smoke-bow:
    uv run train Mjlab-PoliteBow-Flat-MicroDuck \
        --env.scene.num-envs 64 --agent.max_iterations 5 \
        --gpu-ids None --agent.logger tensorboard

# l4x1 @ $0.80/h; ~1000 iters is roughly $0.50. Ctrl-C detaches without killing
# the job — manage it afterwards with `hf jobs ps -a` / `hf jobs cancel <id>`.
# save-interval 100 overrides the cfg's 250 so a timeout or a cancel loses at
# most 99 iterations; the uploader pushes each checkpoint to the HF model repo
# within 60 s and wandb keeps a copy for `resume-bow`. The timeout is the cost
# ceiling, not just a safety net: 2 h on l4x1 caps the run at $1.60.

# Train the polite bow on HF Jobs (4096 envs).
train-bow iters="1000" timeout="2h":
    uv run train Mjlab-PoliteBow-Flat-MicroDuck \
        --env.scene.num-envs 4096 --agent.max_iterations {{iters}} \
        --agent.save-interval 100 \
        --hf-jobs --namespace {{hf_namespace}} --timeout {{timeout}}

# e.g. `just resume-bow abc123xy 2000`, where run_id is the wandb run id (the
# trailing part of the run URL). Resume MUST go through wandb here: the job
# tarball is built from `git ls-files`, so the gitignored logs/ dir is absent
# inside a fresh job and --agent.load-run would find nothing to load.

# Continue a polite-bow run on HF Jobs from its latest wandb checkpoint.
resume-bow run_id iters="1000" timeout="2h":
    uv run train Mjlab-PoliteBow-Flat-MicroDuck \
        --env.scene.num-envs 4096 --agent.max_iterations {{iters}} \
        --agent.save-interval 100 \
        --agent.resume True \
        --wandb-run-path {{wandb_entity}}/mjlab_microduck/{{run_id}} \
        --hf-jobs --namespace {{hf_namespace}} --timeout {{timeout}}

# Pulls the checkpoint from wandb, exports with the obs normalizer BAKED IN
# (scripts/export.py is the only safe path — in-sim play applies the normalizer
# itself and so hides a hand-converted checkpoint's bug), then runs the CPU
# MuJoCo deployment rehearsal with the official walking policy in the other slot.
# The bow rides the ground-pick slot, so press G in the viewer to trigger it;
# the period must match BOW_PERIOD in the env cfg.

# Export a polite-bow checkpoint from wandb and watch it in the local sim.
sim-bow run_id checkpoint="":
    #!/usr/bin/env bash
    set -euo pipefail
    D=$({{just_executable()}} policies)
    CKPT_ARG=""
    [ -n "{{checkpoint}}" ] && CKPT_ARG="--checkpoint {{checkpoint}}"
    uv run scripts/export.py Mjlab-PoliteBow-Flat-MicroDuck \
        --wandb-run-path {{wandb_entity}}/mjlab_microduck/{{run_id}} \
        $CKPT_ARG --num-envs 1
    mv output.onnx bow.onnx
    # macOS forces mujoco.viewer.launch_passive to run under `mjpython`.
    PY=$([ "$(uname -s)" = "Darwin" ] && echo mjpython || echo python)
    uv run "$PY" scripts/infer_policy.py \
        --new-cmd-obs \
        --walking            "$D/alpha_walking.onnx" \
        --ground-pick        bow.onnx \
        --ground-pick-period 4.0
