#!/bin/bash -l
#SBATCH --job-name=train-bart-generation
#SBATCH -t 20:00:00                  # estimated time # TODO: adapt to your needs
#SBATCH -p grete                     # the partition you are training on (i.e., which nodes), for nodes see sinfo -p grete:shared --format=%N,%G
#SBATCH -G A100:1                    # take 1 GPU, see https://docs.hpc.gwdg.de/compute_partitions/gpu_partitions/index.html for more options
#SBATCH --mem-per-gpu=8G             # setting the right constraints for the splitted gpu partitions
#SBATCH --nodes=1                    # total number of nodes
#SBATCH --ntasks=1                   # total number of tasks
#SBATCH --cpus-per-task=8            # number cores per task
#SBATCH --mail-type=all              # send mail when job begins and ends
#SBATCH --mail-user=c.wegner01@stud.uni-goettingen.de
#SBATCH --output=./slurm_files/slurm-%x-%j.out     # where to write output, %x give job name, %j names job id
#aSBATCH --error=./slurm_files/slurm-%x-%j.err      # where to write slurm error

#module load anaconda3
source activate dnlp # Or whatever you called your environment.

pip install --user spacy
pip install --user peft

# For debugging purposes.
python --version
python -m torch.utils.collect_env 2> /dev/null
python -m spacy download en_core_web_sm
