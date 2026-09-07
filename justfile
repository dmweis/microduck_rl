# Task runner for mjlab-microduck. `just` with no arguments lists recipes.

policies_repo := "pollen-robotics/microduck-policies"

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
