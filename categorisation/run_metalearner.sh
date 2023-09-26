#!/bin/bash -l
#SBATCH -o ./logs/%A.out
#SBATCH -e ./logs/%A.err
#SBATCH --time=24:00:00
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=18
#SBATCH --mail-type=ALL
#SBATCH --mail-user=akshay.jagadish@tuebingen.mpg.de

cd ~/vanilla-llama/categorisation/

module purge
module load anaconda/3/2021.11
module load gcc/11 impi/2021.6
module load cuda/11.6
module load pytorch_distributed/gpu-cuda-11.6/1.13.0
pip3 install --user accelerate openai gym ipdb transformers tensorboard
pip3 install --user fire sentencepiece ipdb accelerate tqdm
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

#python rl2/train.py --num-episodes 100000 --save-every 1000 --print-every 100 --test --synthetic --max-steps 96
#python rl2/train.py --num-episodes 500000 --save-every 1000 --print-every 100 --max-steps 70 --env-name claude_generated_tasks_paramsNA_dim3_data100_tasks14000 --first-run-id 0 --noise 0.0
#python rl2/train.py --num-episodes 1000000 --save-every 100 --print-every 100 

## uniform data
# python rl2/train.py --num-episodes 500000 --save-every 1000 --print-every 100 --max-steps 70 --env-name claude_generated_tasks_paramsNA_dim3_data100_tasks14000 --first-run-id 0 --noise 0.0 --synthetic

# prompt version 1
# python rl2/train.py --num-episodes 500000 --save-every 1000 --print-every 100 --max-steps 70 --env-name claude_generated_tasks_paramsNA_dim3_data100_tasks14000_pversion1 --first-run-id 0 --noise 0.0 
# python rl2/train.py --num-episodes 500000 --save-every 1000 --print-every 100 --max-steps 70 --env-name claude_generated_tasks_paramsNA_dim3_data100_tasks14000_pversion1 --first-run-id 0 --noise 0.0 --shuffle

python rl2/train.py --num-episodes 100 --save-every 10 --print-every 100 --env-name claude_generated_tasks_paramsNA_dim3_data100_tasks14000 --first-run-id 0 --noise 0.0 --synthetic