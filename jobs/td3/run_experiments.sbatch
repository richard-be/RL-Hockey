#!/bin/bash
CONFIG_DIR="src/td3/configs"

mkdir -p "$LOG_DIR"
launch_script=jobs/td3/launch_single_config.sbatch

ls "$CONFIG_DIR"/*.yaml
for config in "$CONFIG_DIR"/*.yaml; do
    name=$(basename "$config" .yaml)

    echo "Launching $name"

    export CURRENT_CONFIG="$config"
    sbatch "$launch_script"
done