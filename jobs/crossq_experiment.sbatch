#!/bin/bash
#SBATCH --job-name=RL_Hockey_GG
# give it any name you want
#SBATCH --cpus-per-task=16
# max 24 per node
#SBATCH --partition=day
# choose out of day, week, month depending on job duration
#SBATCH --mem-per-cpu=3G
# max 251GB per node
#SBATCH --gres=gpu:4
# how many gpus to use
# each node has 4 gpus
#SBATCH --time=23:00:00
# job length: the job will run either until completion or until this timer runs out
#SBATCH --error=outputs/crossq/job.%J.err
# %J is the job ID, errors will be written to this file
#SBATCH --output=outputs/crossq/job.%J.out
# the output will be written in this file
#SBATCH --mail-type=ALL
# write a mail if a job begins, ends, fails, gets requeued or stages out
# options: NONE, BEGIN, END, FAIL, REQUEUE, ALL
#SBATCH --mail-user=giorgi.gogelashvili@student.uni-tuebingen.de
# your email
# here will be your commands for running the script


run_one_seed () {
  local seed="$1"
  local gpu="$2"

  local run_dir="outputs/crossq/job_${SLURM_JOB_ID}_seed_${seed}"
  local log_file="${run_dir}/train.log"
  mkdir -p "$run_dir"

  echo "[seed=$seed] starting on GPU=$gpu at $(date)"

  singularity exec \
    --nv \
    --bind "$PROJECT_DIR:/workspace" \
    "$CONTAINER" \
    bash -lc "
      export CUDA_VISIBLE_DEVICES=${gpu}
      python3 /workspace/src/crossq/train.py \
        seed=${seed} \
        hydra.run.dir=/workspace/${run_dir} \
        env.env_id='Hockey-v0' \
        agent_config.actor_lr=0.0003 \
        agent_config.target=False \
        agent_config.soft_l2=False \
        agent_config.q_lr=0.0003 \
        agent_config.weight_norm=True \
        agent_config.utd=1 \
        agent_config.batch_norm_type='BN' \
        env.play_against_latest_model_ratio=0
    " 2>&1 | tee "$log_file"
}

# --- run seeds in batches of NUM_GPUS ---
i=0
while [ $i -lt ${#SEEDS[@]} ]; do
  pids=()

  # launch up to NUM_GPUS experiments in parallel
  for g in "${!GPU_POOL[@]}"; do
    idx=$((i + g))
    [ $idx -ge ${#SEEDS[@]} ] && break

    seed="${SEEDS[$idx]}"
    gpu="${GPU_POOL[$g]}"

    run_one_seed "$seed" "$gpu" &
    pids+=($!)
  done

  # wait for this wave to finish
  fail=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      fail=1
    fi
  done

  if [ $fail -ne 0 ]; then
    echo "Some seeds failed in this wave. Continuing to next wave..."
  fi

  i=$((i + NUM_GPUS))
done

echo "All seeds finished at $(date)"