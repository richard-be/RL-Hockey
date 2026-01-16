#!/bin/bash
#SBATCH --job-name=RL_Hockey_Below
# give it any name you want
#SBATCH --cpus-per-task=4
# max 24 per node
#SBATCH --partition=day
# choose out of day, week, month depending on job duration
#SBATCH --mem-per-cpu=3G
# max 251GB per node
#SBATCH --gres=gpu:4
# how many gpus to use
# each node has 4 gpus
#SBATCH --time=23:55:00
# job length: the job will run either until completion or until this timer runs out
#SBATCH --error=outputs/sac/job.%J.err
# %J is the job ID, errors will be written to this file
#SBATCH --output=outputs/sac/job.%J.out
# the output will be written in this file
#SBATCH --mail-type=ALL
# write a mail if a job begins, ends, fails, gets requeued or stages out
# options: NONE, BEGIN, END, FAIL, REQUEUE, ALL
#SBATCH --mail-user=belowrichard@web.de
# your email
# here will be your commands for running the script


alphas=(0.0 0.1 0.2 0.3)
betas=(0 1 2)
combinations=()
for b in "${betas[@]}"; do
    for a in "${alphas[@]}"; do
        combinations+=("$b $a false")
    done
    combinations+=("$b $a True")
done

i=0
gpu_ids=(0 1 2 3)
max_concurrent=4

for PARAM in "${combinations[@]}"; do
    read beta alpha autotune <<< "$PARAM"
    
    export CUDA_VISIBLE_DEVICES="${gpu_ids[i % 4]}"
    i=$((i+1))
    cmd=(singularity exec \
      --nv \
      --bind $PWD:/workspace \
      $PWD/singularity/images/rl_hockey.simg \
      python3 /workspace/src/sac/run_training.py \
      --alpha "$alpha" --beta "$beta" --total-timesteps 400000
    )

    if [[ "$autotune" == "false" ]]; then
        cmd+=(--no-autotune)
    fi

    echo "Running experiment: alpha=$alpha beta=$beta autotune=$autotune"
    "${cmd[@]}" &

    # throttle: wait every 4 jobs
    if (( i % max_concurrent == 0 )); then
        wait
    fi

done
wait



