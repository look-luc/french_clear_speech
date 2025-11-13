#SBATCH --time=01:00:00
#SBATCH --qos=blanca-clearlab2
#SBATCH --account=blanca-clearlab2
#SBATCH --partition=blanca-clearlab2
#SBATCH --ntasks-per-node=16
#SBATCH --gres=gpu:1
#SBATCH --job-name=regression_model_training
#SBATCH --export=NONE

module purge
module load python/3.14.0

python nasal_nn.py
