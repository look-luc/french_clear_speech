#SBATCH --time=01:00:00        # Job run time (e.g., 1 hour)
#SBATCH --qos=normal           # Quality of Service (e.g., normal, preemptable)
#SBATCH --job-name=regression_model_training # Name of your job
#SBATCH --nodes=1              # Number of nodes to request
#SBATCH --ntasks=1             # Number of tasks (cores) per node
#SBATCH --output=my_python_job.%j.out # Standard output file
#SBATCH --error=my_python_job.%j.err  # Standard error file

# Load the Python module
module purge
module load python/3.14.0 # Replace with the desired Python version (e.g., python/3.9.5)

# Run your Python script
python nasal_nn.py
