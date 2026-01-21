#!/bin/bash
#SBATCH --job-name=RL_Hockey_GG
# give it any name you want
#SBATCH --cpus-per-task=4
# max 24 per node
#SBATCH --partition=test
# choose out of day, week, month depending on job duration
#SBATCH --mem-per-cpu=3G
# max 251GB per node
#SBATCH --gres=gpu:1
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

singularity exec \
  --nv \
  --bind $PWD:/workspace \
  $PWD/singularity/images/rl_hockey.simg \
  python3 /workspace/src/crossq/train.py \
    env.env_id="Hockey-v0" 

