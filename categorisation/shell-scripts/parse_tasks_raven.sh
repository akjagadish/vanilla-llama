#!/bin/bash -l
#SBATCH -o ./logs/%A.out
#SBATCH -e ./logs/%A.err
#SBATCH --time=24:00:00
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:a100:4
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
pip3 install --user accelerate openai gym ipdb transformers tensorboard
pip3 install --user fire sentencepiece ipdb accelerate tqdm
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

python parse_generated_tasks.py  --gpt claude --dims 3 --num-data 100 --tasks 2000 --runs 0  --prompt_version 4 --use_generated_tasklabels --proc_ids "{2000: [0]}"

# path = '/raven/u/ajagadish/vanilla-llama/categorisation/data'#archive
# gpt = 'claude' #'gpt4' #'llama'
# models = ['NA'] #['65B'] #['NA']
# dims = [3]
# num_data_points = [100] #[8]
# tasks = [2000]#[100, 300] #[1000, 2000, 500, 1500] #[20, 10] #[1000, 2000, 500, 1500]
# runs = [0] #{1000: 0, 2000: 0}
# proc_ids = {2000: [0]} #{100: [0, 2], 300: [3]} #{2000: [0, 1, 2, 3, 4, 5, 6]}
# prompt_version = 4
# use_generated_tasklabels = True