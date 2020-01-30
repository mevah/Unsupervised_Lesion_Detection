#!/bin/bash
#$ -o train_VAE_seg_changedloss
#$ -S /bin/bash
#$ -j y
#$ -V
#$ -cwd
#$ -l gpu=1
#$ -l h_vmem=40G
#$ -q gpu.24h.q 
source /home/himeva/.bashrc

#source ~/.bashrc
source /scratch_net/biwidl201/himeva/anaconda2/etc/profile.d/conda.sh
conda activate seg
python vae_training_seg.py
