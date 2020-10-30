#!/bin/bash
experiment=$1
num_runs=$2
slurm=$3
overwrite=$4

conda activate audneurorl

if [[ "$slurm" == "true" ]]
then
    echo "comencing slurm batch parallel across "$slurm" machines"

    $train_cmd --mem 8GB --time 01-00:00:00 JOB=1:$num_jobs ../../data/$experiment/log/main_game.JOB.log batch_individual $overwrite || exit 1; 
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


