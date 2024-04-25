#!/bin/bash

EXPERIMENT_TITLE=$1

echo "Creating experiment directory /home/jmackey/scratch/dynamics_modification/results/${EXPERIMENT_TITLE}"
mkdir "/home/jmackey/scratch/dynamics_modification/results/${EXPERIMENT_TITLE}"

MODEL_NAMES=( "ViT-B-32" "ViT-B-16" )
DATASET_NAMES=( "MNIST" "SVHN" "CIFAR10" "CIFAR100" )
LR_VALS=( 1e-5 )
FT_METHODS=( "full" )

for MODEL_NAME in "${MODEL_NAMES[@]}";
do
    for DATASET_NAME in "${DATASET_NAMES[@]}";
    do
        for LR in "${LR_VALS[@]}";
        do
            for FT_METHOD in "${FT_METHODS[@]}";
            do
                SBATCH_COMMAND="/home/jmackey/scratch/dynamics_modification/scripts/single_job.sh \
                    ${MODEL_NAME} \
                    ${DATASET_NAME} \
                    ${LR} \
                    ${FT_METHOD} \
                    /home/jmackey/scratch/dynamics_modification/results/${EXPERIMENT_TITLE}"
                echo "Running sbatch command ${SBATCH_COMMAND}"
                sbatch ${SBATCH_COMMAND}
            done
        done
    done
done

echo "Experiments submitted!"
            

