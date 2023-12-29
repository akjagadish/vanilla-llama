#!/bin/bash -l
#SBATCH -o ./logs/%A.out
#SBATCH -e ./logs/%A.err
#SBATCH --job-name=Preprocessing
#SBATCH --mail-type=ALL
#SBATCH --mail-user=akshaykjagadish@gmail.com
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=18


cd ~/vanilla-llama/categorisation/data/human/
module purge
module load anaconda/3/2023.03
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

cd devraj2022rational/
python preprocess_data.py

# cd badham2017deficits/
# python preprocess_data.py