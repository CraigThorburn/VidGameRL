#!/bin/bash
#SBATCH --qos=shallow
#SBATCH --mem=4GB
#SBATCH --time=02-00:00:00
#SBATCH --output=/fs/clip-realspeech/projects/vid_game/data/control_game/log/batch.txt
#SBATCH --mail-type=all
#SBATCH --mail-user=craigtho@umiacs.umd.edu



experiment=$1
num_runs=$2
slurm=$3
overwrite=$4

. path.sh
export train_cmd="slurm.pl --config conf/slurm.conf"


source activate audneurorl

if [[ "$slurm" == "true" ]]
then
    echo "comencing slurm batch parallel across "$num_runs" machines"

    $train_cmd --mem 16GB --time 01-00:00:00 JOB=1:$num_runs ../../data/$experiment/log/main_game.JOB.log  run_batch_individual.sh $overwrite || exit 1; 
    wait
    echo "finished"

        echo "not implemented"
elif [[ "$slurm" == "false" ]]
then
        echo "comencing batch without slurm"
	for i in $(eval echo "{1..$num_runs}")
        do
                echo "---------------------"
                echo "launching run "$i
                python main_game.py $host -overwrite $overwrite -modelname run$i
        echo "finished"
        done
else
  echo "problem"
fi


