#!/bin/bash -l
#SBATCH -o ./logs/%A.out
#SBATCH -e ./logs/%A.err
#SBATCH --time=24:00:00
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=18
#SBATCH --mail-type=ALL
#SBATCH --mail-user=akshaykjagadish@gmail.com

cd ~/vanilla-llama/categorisation/

module purge
module load anaconda/3/2021.11
module load gcc/11 impi/2021.6
module load cuda/11.6
module load pytorch_distributed/gpu-cuda-11.6/1.13.0
# pip3 install --user accelerate openai gym ipdb transformers tensorboard
# pip3 install --user fire sentencepiece ipdb accelerate tqdm
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# python rl2/train.py --num-episodes 100000 --save-every 1000 --print-every 100 --test --synthetic --max-steps 96
# python rl2/train.py --num-episodes 500000 --save-every 1000 --print-every 100 --max-steps 70 --env-name claude_generated_tasks_paramsNA_dim3_data100_tasks14000 --first-run-id 0 --noise 0.0
# python rl2/train.py --num-episodes 1000000 --save-every 100 --print-every 100 

## uniform data
# python rl2/train.py --num-episodes 500000 --save-every 1000 --print-every 100 --max-steps 70 --env-name claude_generated_tasks_paramsNA_dim3_data100_tasks14000 --first-run-id 0 --noise 0.0 --synthetic

# prompt version 1
# python rl2/train.py --num-episodes 500000 --save-every 1000 --print-every 100 --max-steps 70 --env-name claude_generated_tasks_paramsNA_dim3_data100_tasks14000_pversion1 --first-run-id 0 --noise 0.0 
# python rl2/train.py --num-episodes 500000 --save-every 1000 --print-every 100 --max-steps 70 --env-name claude_generated_tasks_paramsNA_dim3_data100_tasks14000_pversion1 --first-run-id 0 --noise 0.0 --shuffle

# python rl2/train.py --num-episodes 100 --save-every 10 --print-every 100 --env-name claude_generated_tasks_paramsNA_dim3_data100_tasks14000 --first-run-id 0 --noise 0.0 --synthetic
# python rl2/train_transformer.py --num-episodes 10000 --save-every 10 --print-every 100 --max-steps 70 --env-name tranformer_metalearning_synthetic --first-run-id 0 --noise 0.0 --synthetic --model-name transformer
# python rl2/train_transformer.py --num-episodes 500000 --save-every 10 --print-every 100 --max-steps 70 --env-name tranformer_metalearning_synthetic --first-run-id 0 --noise 0.0 --synthetic --model-name transformer --num_hidden 128 --num_layers 1 --d_model 256 --num_head 2

# python rl2/train_transformer.py --num-episodes 500000 --save-every 1000 --print-every 100 --max-steps 200 --env-name transformer_metalearning_synthetic  --first-run-id 0 --noise 0.0 --model-name transformer --synthetic --num_hidden 256 --num_layers 6 --d_model 64 --num_head 8
# python rl2/train_transformer.py --num-episodes 500000 --save-every 100 --print-every 100 --max-steps 200 --env-name transformer_metalearning_synthetic  --first-run-id 0 --noise 0.0 --model-name transformer --synthetic --num_hidden 256 --num_layers 6 --d_model 64 --num_head 8

python rl2/train_transformer.py --num-episodes 500000 --save-every 100 --print-every 100 --max-steps 250 --env-name claude_generated_tasks_paramsNA_dim3_data100_tasks14000_pversion1  --first-run-id 0 --noise 0.0 --model-name transformer --num_hidden 256 --num_layers 6 --d_model 64 --num_head 8 --batch_size 64 --shuffle