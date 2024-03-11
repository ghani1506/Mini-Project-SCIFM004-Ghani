                  
#!/bin/bash

#SBATCH --job-name=ghani_job
#SBATCH --partition=test
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=12
#SBATCH --cpus-per-task=1
#SBATCH --time=0:0:10
#SBATCH --mem-per-cpu=100M
#SBATCH --account=#SBATCH --ntasks-per-node=3          
#SBATCH --cpus-per-task=1            
#SBATCH --time=1:00:00               

