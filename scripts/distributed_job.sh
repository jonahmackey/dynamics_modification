#!/bin/bash
#SBATCH --account=def-papyan
#SBATCH --mail-user=jonah.mackey@mail.utoronto.ca
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --gpus-per-node=v100:2
#SBATCH --ntasks-per-node=1
#SBATCH --mem=16G
#SBATCH --time=0-00:30

MODEL_NAME=$1
DATASET_NAME=$2
LR=$3
FT_METHOD=$4
RESULTS_PATH=$5

mkdir "${RESULTS_PATH}/${MODEL_NAME}_${DATASET_NAME}_${FT_METHOD}_lr=${LR}_id=${SLURM_JOB_ID}"
module load python/3.10
module load scipy-stack
source /home/jmackey/dm/bin/activate

echo "Job Array ID / Job ID: $SLURM_ARRAY_JOB_ID / $SLURM_JOB_ID"

python /home/jmackey/scratch/dynamics_modification/main_dist.py \
    --model_name ${MODEL_NAME} \
    --dataset_name ${DATASET_NAME} \
    --data_path /home/jmackey/datasets \
    --lr ${LR} \
    --batch_size 128 \
    --num_iters 1000 \
    --warmup_steps 200 \
    --print_every 100 \
    --ft_method ${FT_METHOD} \
    --heads_path /home/jmackey/scratch/dynamics_modification/heads \
    --results_path "${RESULTS_PATH}/${MODEL_NAME}_${DATASET_NAME}_${FT_METHOD_}lr=${LR}_id=${SLURM_JOB_ID}" \
    --job_id "${SLURM_JOB_ID}" \
    --world_size 2