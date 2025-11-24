#!/bin/bash

# Training script for CIFAR-100 with OFQ + SQuaT (DeiT-Small, W1A1: 1-bit weight, 1-bit activation)

CONFIG_FILE="configs/cifar100_deit_tiny_squat.yml"
DATA_DIR="./data/CIFAR100"

python train_squat.py \
    --config ${CONFIG_FILE} \
    ${DATA_DIR} \
    --dataset torch/cifar100 \
    --model deit_small_distilled_patch16_224 \
    --num-classes 100 \
    --img-size 224 \
    --batch-size 128 \
    --epochs 300 \
    --opt adamw \
    --lr 0.001 \
    --weight-decay 0.05 \
    --sched cosine \
    --warmup-lr 0.0001 \
    --min-lr 0.00001 \
    --warmup-epochs 10 \
    --wq-enable \
    --wq-mode statsq \
    --wq-bitw 1 \
    --aq-enable \
    --aq-mode lsq \
    --aq-bitw 1 \
    --qmodules blocks \
    --use-squat \
    --QFeatureFlag \
    --feature-levels 1 \
    --use-adaptor \
    --use-student-quant-params \
    --kd-T 4.0 \
    --kd-beta 1.0 \
    --kd-gamma 1.0 \
    --feature-distill-loss-type L2 \
    --seed 42 \
    --gpu-id 0 \
    --workers 4




