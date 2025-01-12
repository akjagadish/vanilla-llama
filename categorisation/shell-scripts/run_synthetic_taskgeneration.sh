#!/bin/bash -l
#SBATCH -o ./logs/%A.out
#SBATCH -e ./logs/%A.err
#SBATCH --job-name=SyntheticTaskGeneration  
#SBATCH --mail-type=ALL
#SBATCH --mail-user=akshaykjagadish@gmail.com
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=18


cd ~/vanilla-llama/categorisation/rl2/

module purge
module load anaconda/3/2023.03
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# python simulate_data.py --num-tasks 5000 --num-dims 3 --max-steps 100
# python simulate_data.py --num-tasks 5000 --num-dims 3 --max-steps 100  --nonlinear
python simulate_data.py --num-tasks 10 --num-dims 3 --max-steps 100  --rmc