#!/bin/bash
#SBATCH --account=def-papyan
#SBATCH --mail-user=jonah.mackey@mail.utoronto.ca
#SBATCH --mail-type=ALL
#SBATCH --gpus-per-node=v100:1
#SBATCH --mem=16G
#SBATCH --time=0-03:00

MODEL_NAME=$1
DATASET_NAME=$2
LR=$3
FT_METHOD=$4
RESULTS_PATH=$5

mkdir "${RESULTS_PATH}/${MODEL_NAME}_${DATASET_NAME}_${FT_METHOD}_id=${SLURM_JOB_ID}"
module load python/3.10
module load scipy-stack
source /home/jmackey/dm/bin/activate

echo "Job Array ID / Job ID: $SLURM_ARRAY_JOB_ID / $SLURM_JOB_ID"

python /home/jmackey/scratch/dynamics_modification/main.py \
    --model_name ${MODEL_NAME} \
    --dataset_name ${DATASET_NAME} \
    --data_path /home/jmackey/scratch/dynamics_modification/datasets \
    --lr ${LR} \
    --batch_size 128 \
    --num_iters 3000 \
    --warmup_steps 200 \
    --weight_decay 0.0 \
    --print_every 100 \
    --ft_method ${FT_METHOD} \
    --heads_path /home/jmackey/scratch/dynamics_modification/heads \
    --results_path "${RESULTS_PATH}/${MODEL_NAME}_${DATASET_NAME}_${FT_METHOD}_id=${SLURM_JOB_ID}" \
    --local_path "${SLURM_TMPDIR}" \
    --job_id "${SLURM_JOB_ID}"