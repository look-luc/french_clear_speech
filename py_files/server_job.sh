#SBATCH --time=01:00:00        # Job run time (e.g., 1 hour)
#SBATCH --qos=blanca-clearlab2  # Quality of Service (e.g., normal, preemptable)
#SBATCH --account=blanca-clearlab2
#SBATCH --partition=blanca-clearlab2
#SBATCH --ntasks-per-node=16
#SBATCH --gres=gpu:1
#SBATCH --job-name=regression_model_training
#SBATCH --export=NONE

# Load the Python module
module purge
module load python/3.14.0 # Replace with the desired Python version (e.g., python/3.9.5)

# Run your Python script
python nasal_nn.py
