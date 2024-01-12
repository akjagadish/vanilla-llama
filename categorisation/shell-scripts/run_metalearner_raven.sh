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



## ermi model for 4 dims with parameters matched to pfn model
# python rl2/train_transformer.py --first-run-id 0 --sample-to-match-max-steps --num-episodes 500000 --max-steps 300 --num-dims 4 --env-name  claude_generated_tasks_paramsNA_dim4_data650_tasks8950_pversion5_stage1 --save-every 100 --print-every 100  --noise 0.0 --model-name transformer --num_hidden 1024 --num_layers 12 --d_model 512 --num_head 4 --batch_size 32 --shuffle --env-dir u/ajagadish/vanilla-llama/categorisation/data --shuffle-features --lr .001
# python rl2/train_transformer.py --first-run-id 0 --sample-to-match-max-steps --num-episodes 500000 --max-steps 300 --num-dims 4 --env-name  claude_generated_tasks_paramsNA_dim4_data650_tasks8950_pversion5_stage1 --save-every 100 --print-every 100  --noise 0.0 --model-name transformer --num_hidden 1024 --num_layers 12 --d_model 512 --num_head 4 --batch_size 32 --shuffle --env-dir u/ajagadish/vanilla-llama/categorisation/data --shuffle-features --lr .001  --restart-training #--restart-episode-id 246815
## lr .0001
# python rl2/train_transformer.py --first-run-id 0 --sample-to-match-max-steps --num-episodes 500000 --max-steps 300 --num-dims 4 --env-name  claude_generated_tasks_paramsNA_dim4_data650_tasks8950_pversion5_stage1 --save-every 100 --print-every 100  --noise 0.0 --model-name transformer --num_hidden 1024 --num_layers 12 --d_model 512 --num_head 4 --batch_size 32 --shuffle --env-dir u/ajagadish/vanilla-llama/categorisation/data --shuffle-features --lr .0001
# python rl2/train_transformer.py --first-run-id 0 --sample-to-match-max-steps --num-episodes 500000 --max-steps 300 --num-dims 4 --env-name  claude_generated_tasks_paramsNA_dim4_data650_tasks8950_pversion5_stage1 --save-every 100 --print-every 100  --noise 0.0 --model-name transformer --num_hidden 1024 --num_layers 12 --d_model 512 --num_head 4 --batch_size 32 --shuffle --env-dir u/ajagadish/vanilla-llama/categorisation/data --shuffle-features --lr .0001 --restart-training #--restart-episode-id 246957
## lr .0003
# python rl2/train_transformer.py --first-run-id 0 --sample-to-match-max-steps --num-episodes 500000 --max-steps 300 --num-dims 4 --env-name  claude_generated_tasks_paramsNA_dim4_data650_tasks8950_pversion5_stage1 --save-every 100 --print-every 100  --noise 0.0 --model-name transformer --num_hidden 1024 --num_layers 12 --d_model 512 --num_head 4 --batch_size 32 --shuffle --env-dir u/ajagadish/vanilla-llama/categorisation/data --shuffle-features --lr .0003
python rl2/train_transformer.py --first-run-id 0 --sample-to-match-max-steps --num-episodes 500000 --max-steps 300 --num-dims 4 --env-name  claude_generated_tasks_paramsNA_dim4_data650_tasks8950_pversion5_stage1 --save-every 100 --print-every 100  --noise 0.0 --model-name transformer --num_hidden 1024 --num_layers 12 --d_model 512 --num_head 4 --batch_size 32 --shuffle --env-dir u/ajagadish/vanilla-llama/categorisation/data --shuffle-features --lr .0003 --restart-training #--restart-episode-id 247302

### synthetic data
# python rl2/train.py --num-episodes 500000 --save-every 1000 --print-every 100 --max-steps 70 --env-name claude_generated_tasks_paramsNA_dim3_data100_tasks14000 --first-run-id 0 --noise 0.0 --synthetic
# python rl2/train_transformer.py --synthetic --nonlinear --num-episodes 10000 --save-every 100 --print-every 10 --max-steps 250 --env-name claude_generated_tasks_paramsNA_dim3_data100_tasks11518_pversion4 --first-run-id 0 --noise 0.0 --model-name transformer --num_hidden 256 --num_layers 6 --d_model 64 --num_head 8 --batch_size 64 --shuffle --first-run-id 0
# dim 3 synthetic data
# (done) python rl2/train_transformer.py  --synthetic --num-episodes 500000 --save-every 100 --print-every 10 --max-steps 400 --num-dims 3  --env-name dim3synthetic --first-run-id 0 --noise 0.0 --model-name transformer --num_hidden 256 --num_layers 6 --d_model 64 --num_head 8 --batch_size 64 --shuffle 
# (done) python rl2/train_transformer.py  --synthetic --nonlinear --num-episodes 500000 --save-every 100 --print-every 10 --max-steps 400 --num-dims 3  --env-name dim3synthetic --first-run-id 0 --noise 0.0 --model-name transformer --num_hidden 256 --num_layers 6 --d_model 64 --num_head 8 --batch_size 64 --shuffle 
## dim 6 synthetic data
# (done) python rl2/train_transformer.py  --synthetic --nonlinear --num-episodes 500000 --save-every 100 --print-every 100 --max-steps 400 --num-dims 6 --env-name dim6synthetic --first-run-id 0 --noise 0.0 --model-name transformer --num_hidden 256 --num_layers 6 --d_model 64 --num_head 8 --batch_size 64 --shuffle
# (done) python rl2/train_transformer.py  --synthetic --num-episodes 500000 --save-every 100 --print-every 100 --max-steps 400 --num-dims 6 --env-name dim6synthetic --first-run-id 0 --noise 0.0 --model-name transformer --num_hidden 256 --num_layers 6 --d_model 64 --num_head 8 --batch_size 64 --shuffle
# (ongoing) python rl2/train_transformer.py  --synthetic --nonlinear --num-episodes 500000 --save-every 100 --print-every 100 --max-steps 650 --num-dims 6 --env-name dim6synthetic --first-run-id 1 --noise 0.0 --model-name transformer --num_hidden 256 --num_layers 6 --d_model 64 --num_head 8 --batch_size 64 --shuffle
python rl2/train_transformer.py  --synthetic --num-episodes 500000 --save-every 100 --print-every 100 --max-steps 650 --num-dims 6 --env-name dim6synthetic --first-run-id 1 --noise 0.0 --model-name transformer --num_hidden 256 --num_layers 6 --d_model 64 --num_head 8 --batch_size 64 --shuffle

## llm generated data: prompt version 1
# python rl2/train_transformer.py --num-episodes 500000 --save-every 100 --print-every 100 --max-steps 250 --env-name claude_generated_tasks_paramsNA_dim3_data100_tasks14000_pversion1  --first-run-id 0 --noise 0.0 --model-name transformer --num_hidden 256 --num_layers 6 --d_model 64 --num_head 8 --batch_size 64 --shuffle
# python rl2/train_transformer.py --num-episodes 500000 --save-every 100 --print-every 100 --max-steps 250 --env-name claude_generated_tasks_paramsNA_dim3_data100_tasks14000_pversion1  --first-run-id 0 --noise 0.0 --model-name transformer --num_hidden 256 --num_layers 6 --d_model 64 --num_head 8 --batch_size 64 

## llm generated data: prompt version 4
# python rl2/train_transformer.py --num-episodes 500000 --save-every 100 --print-every 100 --max-steps 250 --env-name claude_generated_tasks_paramsNA_dim3_data100_tasks11518_pversion4  --first-run-id 0 --noise 0.0 --model-name transformer --num_hidden 256 --num_layers 6 --d_model 64 --num_head 8 --batch_size 64 --shuffle
# python rl2/train_transformer.py --num-episodes 500000 --save-every 100 --print-every 100 --max-steps 250 --env-name claude_generated_tasks_paramsNA_dim3_data100_tasks11518_pversion4  --first-run-id 0 --noise 0.0 --model-name transformer --num_hidden 256 --num_layers 6 --d_model 64 --num_head 8 --batch_size 64
