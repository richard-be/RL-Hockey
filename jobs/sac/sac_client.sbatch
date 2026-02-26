#!/bin/bash
#SBATCH --job-name=RL_T
# give it any name you want
#SBATCH --cpus-per-task=4
# max 24 per node
#SBATCH --partition=day
# choose out of day, week, month depending on job duration
#SBATCH --mem-per-cpu=3G
# max 251GB per node
#SBATCH --gres=gpu:1
# how many gpus to use
# each node has 4 gpus
#SBATCH --time=15:00:00
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

singularity exec \
  --nv \
  --bind $PWD:/workspace \
  $PWD/singularity/images/rl_hockey_client.simg \
  bash -c "/workspace/src/client/autorestart.sh \
  --token 49415f80-886d-4325-b0cb-e100d93cdd17 --server-url "comprl.cs.uni-tuebingen.de" --server-port 65335 --args --agent=sac --model=models/sac/sac_2_1_1000000_1771781495.pkl"

