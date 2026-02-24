#!/bin/bash
#SBATCH --job-name=kl_hockey
# give it any name you want
#SBATCH --cpus-per-task=4
# max 24 per node
#SBATCH --partition=day
# choose out of day, week, month depending on job duration
#SBATCH --mem-per-cpu=3G
# max 251GB per node

#SBATCH --time=23:00:00
# job length: the job will run either until completion or until this timer runs out
#SBATCH --error=outputs/td3/job.%J.err
# %J is the job ID, errors will be written to this file
#SBATCH --output=outputs/td3/job.%J.out
# the output will be written in this file
#SBATCH --mail-type=ALL
# write a mail if a job begins, ends, fails, gets requeued or stages out
# options: NONE, BEGIN, END, FAIL, REQUEUE, ALL
#SBATCH --mail-user=jakob.j.kleine@outlook.de
# your email
# here will be your commands for running the script

if [ -z "$CURRENT_CONFIG" ]; then
  echo "Error: CURRENT_CONFIG is empty"
  exit 1
fi

SCRIPT="scripts.td3.train"
echo "Launching config file $CURRENT_CONFIG"

singularity exec \
    --nv \
    --bind $PWD:/workspace \
    $PWD/singularity/images/rl_hockey.simg \
    python3 -m $SCRIPT --config "$CURRENT_CONFIG"