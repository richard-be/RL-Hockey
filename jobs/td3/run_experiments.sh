#!/bin/bash
CONFIG_DIR="experiments/td3/configs/prioritized_replay/rnd"/*.yaml
launch_script=jobs/td3/launch_single_config.sbatch

# ls "$CONFIG_DIR"/**/*.yaml
for config in $CONFIG_DIR; do
    # name=$(basename "$config" .yaml)
    # echo "Launching $name"
    echo "Launching config file $config"
    export CURRENT_CONFIG="$config"
    sbatch "$launch_script"
done