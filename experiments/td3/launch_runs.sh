#!/usr/bin/env bash

CONFIG_DIR="experiments/td3/configs"
SCRIPT="src/td3/td3_cleanrl.py"
LOG_DIR="logs"

mkdir -p "$LOG_DIR"

ls "$CONFIG_DIR"/*.yaml
for config in "$CONFIG_DIR"/*.yaml; do
    name=$(basename "$config" .yaml)

    echo "Launching $name"

    python "$SCRIPT" \
        --config "$config" \
        > "$LOG_DIR/$name.out" \
        2> "$LOG_DIR/$name.out" &
done

wait

echo "All runs finished."
