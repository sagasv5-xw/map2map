#!/bin/bash
#SBATCH --job-name=15_0
#SBATCH --output=%x-%j.out
#SBATCH --partition=TWIG
#SBATCH --nodes=1
#SBATCH --gpus=a100-40:3
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=xiaowen4@andrew.cmu.edu
##SBATCH --wait
hostname; pwd; date

data_root_dir="/hildafs/home/xzhangn/xzhangn/sim_output/dmo-100MPC-train/15_0"
in_dir="dmo-64"
tgt_dir="dmo-512"

train_dirs="set*/output/PART_0*"
style_dirs="style/set*/PART_0*.npy"


srun python /hildafs/projects/phy200018p/xzhangn/sr_pipeline/3.5-training/15_0/map2map/m2m.py train \
    --train-in-patterns "$data_root_dir/$in_dir/$train_dirs/disp.npy","$data_root_dir/$in_dir/$train_dirs/vel.npy" \
    --train-tgt-patterns "$data_root_dir/$tgt_dir/$train_dirs/disp.npy","$data_root_dir/$tgt_dir/$train_dirs/vel.npy" \
    --train-style-pattern "$style_dirs" \
    --in-norms cosmology.dis,cosmology.vel --tgt-norms cosmology.dis,cosmology.vel \
    --augment --aug-shift 64 \
    --crop 12 --crop-step 12 --pad 3 --scale-factor 8 \
    --model styled_srsgan.G --adv-model styled_srsgan.D --cgan --callback-at . \
    --adv-start 1 --adv-wgan-gp-interval 16 \
    --lr 1e-5 --adv-lr 2e-5 --optimizer-args '{"betas": [0, 0.99]}' \
    --batches 1 --loader-workers 16 --load-state checkpoint.pt \
    --epochs 9999 \

date
