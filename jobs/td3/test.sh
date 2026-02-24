SCRIPT="src/td3/td3_cleanrl.py"
CURRENT_CONFIG="src/td3/configs/sp_0.yaml"

singularity exec \
    --nv \
    --bind $PWD:/workspace \
    $PWD/singularity/images/rl_hockey.simg \
    python3 /workspace/$SCRIPT --config "$CURRENT_CONFIG"