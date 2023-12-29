#!/bin/bash -l
#SBATCH -o ./logs/%A.out
#SBATCH -e ./logs/%A.err
#SBATCH --job-name=simulate_metalearner
#SBATCH --mail-type=ALL
#SBATCH --mail-user=akshaykjagadish@gmail.com
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=18


cd ~/vanilla-llama/categorisation/

module purge
module load anaconda/3/2023.03
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}


## johannsen
# python rl2/simulate.py --experiment johanssen_categorisation --num-runs 1 --beta 1.  --model-name env=claude_generated_tasks_paramsNA_dim4_data650_tasks8950_pversion5_stage1_model=transformer_num_episodes500000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_run=0
# mean fitted beta from badham fits
# python rl2/simulate.py --experiment johanssen_categorisation --num-runs 1 --beta 0.1856 --model-name env=claude_generated_tasks_paramsNA_dim4_data650_tasks8950_pversion5_stage1_model=transformer_num_episodes500000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_run=0
# # mean fitted beta from devraj fits
# python rl2/simulate.py --experiment johanssen_categorisation --num-runs 1 --beta 0.0950 --model-name env=claude_generated_tasks_paramsNA_dim4_data650_tasks8950_pversion5_stage1_model=transformer_num_episodes500000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_run=0
# bit higher
python rl2/simulate.py --experiment johanssen_categorisation --num-runs 1 --beta 0.15 --model-name env=claude_generated_tasks_paramsNA_dim4_data650_tasks8950_pversion5_stage1_model=transformer_num_episodes500000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_run=0
# python rl2/simulate.py --experiment johanssen_categorisation --num-runs 1 --beta 0.25 --model-name env=claude_generated_tasks_paramsNA_dim4_data650_tasks8950_pversion5_stage1_model=transformer_num_episodes500000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_run=0
# python rl2/simulate.py --experiment johanssen_categorisation --num-runs 1 --beta 0.3 --model-name env=claude_generated_tasks_paramsNA_dim4_data650_tasks8950_pversion5_stage1_model=transformer_num_episodes500000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_run=0
# python rl2/simulate.py --experiment johanssen_categorisation --num-runs 1 --beta 0.4 --model-name env=claude_generated_tasks_paramsNA_dim4_data650_tasks8950_pversion5_stage1_model=transformer_num_episodes500000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_run=0

# python rl2/simulate.py --experiment smith_categorisation --num-runs 10 --beta 0.  --model-name env=claude_generated_tasks_paramsNA_dim6_data500_tasks12910_pversion5_stage2_model=transformer_num_episodes500000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_run=1
# python rl2/simulate.py --experiment smith_categorisation --num-runs 10 --beta 0.1 --model-name env=claude_generated_tasks_paramsNA_dim6_data500_tasks12910_pversion5_stage2_model=transformer_num_episodes500000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_run=1
# python rl2/simulate.py --experiment smith_categorisation --num-runs 10 --beta 0.2 --model-name env=claude_generated_tasks_paramsNA_dim6_data500_tasks12910_pversion5_stage2_model=transformer_num_episodes500000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_run=1
# python rl2/simulate.py --experiment smith_categorisation --num-runs 10 --beta 0.3 --model-name env=claude_generated_tasks_paramsNA_dim6_data500_tasks12910_pversion5_stage2_model=transformer_num_episodes500000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_run=1
# python rl2/simulate.py --experiment smith_categorisation --num-runs 10 --beta 0.4 --model-name env=claude_generated_tasks_paramsNA_dim6_data500_tasks12910_pversion5_stage2_model=transformer_num_episodes500000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_run=1
# python rl2/simulate.py --experiment smith_categorisation --num-runs 10 --beta 0.5 --model-name env=claude_generated_tasks_paramsNA_dim6_data500_tasks12910_pversion5_stage2_model=transformer_num_episodes500000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_run=1
# python rl2/simulate.py --experiment smith_categorisation --num-runs 10 --beta 0.6 --model-name env=claude_generated_tasks_paramsNA_dim6_data500_tasks12910_pversion5_stage2_model=transformer_num_episodes500000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_run=1
# python rl2/simulate.py --experiment smith_categorisation --num-runs 10 --beta 0.7 --model-name env=claude_generated_tasks_paramsNA_dim6_data500_tasks12910_pversion5_stage2_model=transformer_num_episodes500000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_run=1
# python rl2/simulate.py --experiment smith_categorisation --num-runs 10 --beta 0.8 --model-name env=claude_generated_tasks_paramsNA_dim6_data500_tasks12910_pversion5_stage2_model=transformer_num_episodes500000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_run=1
# python rl2/simulate.py --experiment smith_categorisation --num-runs 10 --beta 0.9 --model-name env=claude_generated_tasks_paramsNA_dim6_data500_tasks12910_pversion5_stage2_model=transformer_num_episodes500000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_run=1
# python rl2/simulate.py --experiment smith_categorisation --num-runs 10 --beta 1.0 --model-name env=claude_generated_tasks_paramsNA_dim6_data500_tasks12910_pversion5_stage2_model=transformer_num_episodes500000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_run=1

# python rl2/simulate.py --experiment shepard_categorisation --model-name env=claude_generated_tasks_paramsNA_dim3_data100_tasks11518_pversion4_model=transformer_num_episodes500000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8_noise0.0_shuffleTrue_run=1

## deprecated
# python rl2/save_metalearner_choices.py  --experiment claude_generated_tasks_paramsNA_dim6_data500_tasks12911_pversion5_stage1_model=transformer_num_episodes500000_num_hidden=256_lr0.0003_num_layers=6_d_model=64_num_head=8 --task-name smith1998