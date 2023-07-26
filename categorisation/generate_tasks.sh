#!/bin/bash -l
#SBATCH -o ./logs/%A.out
#SBATCH -e ./logs/%A.err
#SBATCH --time=24:00:00
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:a100:2
#SBATCH --mem=200G
#SBATCH --cpus-per-task=20
#SBATCH --mail-type=ALL
#SBATCH --mail-user=akshay.jagadish@tuebingen.mpg.de

cd ~/vanilla-llama/categorisation/

module purge
module load anaconda/3/2021.11
module load gcc/11 impi/2021.6
module load cuda/11.6
module load pytorch_distributed/gpu-cuda-11.6/1.13.0
pip3 install --user accelerate openai gym ipdb transformers tensorboard python-dotenv
pip3 install --user fire sentencepiece ipdb accelerate tqdm
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

#python generate_tasks.py --llama-path /ptmp/mbinz/new --model 65B --proc-id 999 --num-tasks 5 --num-data 100 --max-length 4000
#python generate_tasks.py --llama-path /ptmp/mbinz/new --model NA --proc-id 1 --num-tasks 100 --num-data 100 --max-length 2000 --run-gpt gpt3
#python generate_tasks.py --llama-path /ptmp/mbinz/new --model NA --proc-id 2 --num-tasks 100 --num-data 100 --max-length 2000 --run-gpt gpt4
#python generate_tasks.py --llama-path /ptmp/mbinz/new --model NA --proc-id 3 --num-tasks 10 --num-data 100 --max-length 2000 --run-gpt gpt4
#python generate_tasks.py --llama-path /ptmp/mbinz/new --model NA --proc-id 4 --num-tasks 10 --num-data 100 --max-length 2000 --run-gpt gpt4
#python generate_tasks.py --llama-path /ptmp/mbinz/new --model NA --proc-id 5 --num-tasks 10 --num-data 100 --max-length 2000 --run-gpt gpt4
#python generate_tasks.py --llama-path /ptmp/mbinz/new --model NA --proc-id 6 --num-tasks 10 --num-data 100 --max-length 2000 --run-gpt gpt4

#python generate_tasks.py --llama-path /ptmp/mbinz/new --model NA --proc-id 1 --num-tasks 10 --num-data 100 --max-length 2000 --run-gpt gpt4
#python generate_tasks.py --llama-path /ptmp/mbinz/new --model NA --proc-id 2 --num-tasks 10 --num-data 100 --max-length 200 --run-gpt gpt4


#python play.py --llama-path /ptmp/mbinz/new --model 65B --proc-id 999 --num-tasks 5 --num-data 100 --max-length 4000
#python play.py --llama-path /ptmp/mbinz/new --model NA --proc-id 999 --num-tasks 10 --num-data 100 --max-length 200 --run-gpt gpt3
#python play.py --llama-path /ptmp/mbinz/new --model NA --proc-id 999 --num-tasks 10 --num-data 100 --max-length 2000 --run-gpt gpt3

#python generate_tasks.py --llama-path /ptmp/mbinz/new --model NA --proc-id 0 --num-tasks 10 --num-data 100 --max-length 2000 --run-gpt claude
#python generate_tasks.py --llama-path /ptmp/mbinz/new --model NA --proc-id 0 --num-tasks 2000 --num-data 100 --max-length 3000 --run-gpt claude
#python generate_tasks.py --llama-path /ptmp/mbinz/new --model NA --proc-id 1 --num-tasks 2000 --num-data 100 --max-length 3000 --run-gpt claude
#python generate_tasks.py --llama-path /ptmp/mbinz/new --model NA --proc-id 2 --num-tasks 2000 --num-data 100 --max-length 3000 --run-gpt claude
#python generate_tasks.py --llama-path /ptmp/mbinz/new --model NA --proc-id 3 --num-tasks 2000 --num-data 100 --max-length 3000 --run-gpt claude
#python generate_tasks.py --llama-path /ptmp/mbinz/new --model NA --proc-id 4 --num-tasks 2000 --num-data 100 --max-length 3000 --run-gpt claude
#python generate_tasks.py --llama-path /ptmp/mbinz/new --model NA --proc-id 5 --num-tasks 2000 --num-data 100 --max-length 3000 --run-gpt claude
python generate_tasks.py --llama-path /ptmp/mbinz/new --model NA --proc-id 6 --num-tasks 2000 --num-data 100 --max-length 3000 --run-gpt claude


#python generate_tasks.py --llama-path /ptmp/mbinz/new --model NA --proc-id 1 --num-tasks 10 --num-data 100 --max-length 2500 --run-gpt claude
#python generate_tasks.py --llama-path /ptmp/mbinz/new --model NA --proc-id 2 --num-tasks 10 --num-data 100 --max-length 3000 --run-gpt claude
#python generate_tasks.py --llama-path /ptmp/mbinz/new --model NA --proc-id 3 --num-tasks 10 --num-data 100 --max-length 4000 --run-gpt claude