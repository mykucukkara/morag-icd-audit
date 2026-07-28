import yaml
from pathlib import Path

def generate_slurm_script(config: dict, script_name: str, command: str, output_path: str | Path):
    template = f"""#!/bin/bash
#SBATCH --account={config.get('account', '[TRUBA_ACCOUNT]')}
#SBATCH --partition={config.get('partition_gpu', 'barbun-cuda')}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={config.get('cpus_per_task_gpu', 20)}
#SBATCH --gres={config.get('gres', 'gpu:1')}
#SBATCH --time={config.get('time_gpu', '3-00:00:00')}
#SBATCH --job-name={script_name}
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

echo "Starting job on $(date)"
echo "Node: $(hostname)"
nvidia-smi

source ~/miniconda3/etc/profile.d/conda.sh
conda activate {config.get('env_name', 'morag_icd')}

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

{command}

echo "Finished job on $(date)"
"""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(template)
