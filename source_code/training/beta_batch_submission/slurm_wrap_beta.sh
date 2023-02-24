#!/bin/bash

## Name of the job
#SBATCH --job-name=betaFinetune

## std error and std out
##SBATCH --output=./Slurm_beta_finetune/slurm.beta.%j.%N.out
##SBATCH --error=./Slurm_beta_finetune/slurm.beta.%j.%N.err 


## run job on specific node 
#SBATCH --nodelist=hpc-rc11,hpc-rc10   #r on: hpc04,hpc05, hpc06 hpc-rc08

## set run on bigmem node only # can change later
#SBATCH --mem=100gb #400gb
#SBATCH --cpus-per-task=4 #24

## set partition/queue
#SBATCH -p normal 

## set max wallclock time 
##SBATCH --time=30:00
#SBATCH --time=2-00:00:00

## Send notification to mail list
## Check out the email_address before submitting
#SBATCH --mail-user=<email_address>


filename=$1

mkdir -p ./Slurm_beta_finetune

while read beta_num; do
	echo "sending job to slurm for beta factor $beta_num"

	sbatch -o ./Slurm_beta_finetune/slurm.beta.%j.%N.out -e ./Slurm_beta_finetune/slurm.beta.%j.%N.err -p normal -t 2-00:00:00 -c 8 --job-name betaFinetune --nodelist=hpc-rc10 --wrap="./beta_submission_single.sh '$beta_num'" 
	sleep 2

done <$filename





