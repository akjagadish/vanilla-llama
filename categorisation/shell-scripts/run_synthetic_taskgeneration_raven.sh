#!/bin/bash -l
#SBATCH -o ./logs/%A.out
#SBATCH -e ./logs/%A.err
#SBATCH --time=24:00:00
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=120G
#SBATCH --cpus-per-task=18
#SBATCH --mail-type=ALL
#SBATCH --mail-user=akshaykjagadish@gmail.com

cd ~/vanilla-llama/categorisation/rl2

module purge
module load anaconda/3/2021.11
module load gcc/11 impi/2021.6
module load cuda/11.6
module load pytorch_distributed/gpu-cuda-11.6/1.13.0
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# python simulate_data.py --num-tasks 5000 --num-dims 3 --max-steps 100  --rmc
# python simulate_data.py --num-tasks 5000 --num-dims 3 --max-steps 100  --rmc --batch-size 100
#python simulate_data.py --num-tasks 6518 --num-dims 3 --max-steps 100  --rmc --batch-size 100
python simulate_data.py --num-tasks 6518 --num-dims 3 --max-steps 600  --rmc --batch-size 100