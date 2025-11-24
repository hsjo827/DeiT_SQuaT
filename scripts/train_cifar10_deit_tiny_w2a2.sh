#!/bin/bash

# Training script for CIFAR-10 with OFQ + SQuaT (DeiT-Tiny, W2A2: 2-bit weight, 2-bit activation)

CONFIG_FILE="configs/cifar10_deit_tiny_squat.yml"
DATA_DIR="./data/CIFAR10"

python train_squat.py \
    --config ${CONFIG_FILE} \
    ${DATA_DIR} \
    --dataset torch/cifar10 \
    --model deit_tiny_distilled_patch16_224 \
    --num-classes 10 \
    --img-size 224 \
    --batch-size 128 \
    --epochs 100 \
    --opt adamw \
    --lr 0.001 \
    --weight-decay 0.05 \
    --sched cosine \
    --warmup-lr 0.0001 \
    --min-lr 0.00001 \
    --warmup-epochs 10 \
    --wq-enable \
    --wq-mode statsq \
    --wq-bitw 2 \
    --aq-enable \
    --aq-mode lsq \
    --aq-bitw 2 \
    --qmodules blocks \
    --use-squat \
    --QFeatureFlag \
    --feature-levels 2 \
    --use-adaptor \
    --use-student-quant-params \
    --kd-T 4.0 \
    --kd-beta 1.0 \
    --kd-gamma 1.0 \
    --feature-distill-loss-type L2 \
    --seed 42 \
    --gpu-id 0 \
    --workers 4




