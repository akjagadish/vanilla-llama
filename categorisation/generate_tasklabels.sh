#!/bin/bash -l
#SBATCH -o ./logs/%A.out
#SBATCH -e ./logs/%A.err
#SBATCH --time=24:00:00
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:a100:2
#SBATCH --mem=200G
#SBATCH --cpus-per-task=20
#SBATCH --mail-type=ALL
#SBATCH --mail-user=akshaykjagadish@gmail.com

cd ~/vanilla-llama/categorisation/

module purge
module load anaconda/3/2021.11
module load gcc/11 impi/2021.6
module load cuda/11.6
module load pytorch_distributed/gpu-cuda-11.6/1.13.0
pip3 install --user accelerate openai gym ipdb transformers tensorboard python-dotenv
pip3 install --user fire sentencepiece ipdb accelerate tqdm
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

## generate task labels
# python generate_tasklabels.py --model NA --proc-id 0 --num-runs 100 --num-tasks 250 --num-dim 3 --max-length 10000 --run-gpt claude --prompt-version 5 
## pool generated task labels
python generate_tasklabels.py --model NA --proc-id 0 --num-runs 100 --num-tasks 250 --num-dim 3 --max-length 10000 --run-gpt claude --prompt-version 5 --pool
