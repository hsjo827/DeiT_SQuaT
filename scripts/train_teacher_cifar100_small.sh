#!/bin/bash

# Training script for Teacher Model on CIFAR-100 (DeiT-Small, Full-precision, no quantization)

CONFIG_FILE="configs/cifar100_teacher_small.yml"
DATA_DIR="./data/CIFAR100"

python train_teacher.py \
    --config ${CONFIG_FILE} \
    ${DATA_DIR} \
    --dataset torch/cifar100 \
    --model deit_small_distilled_patch16_224 \
    --num-classes 100 \
    --img-size 224 \
    --batch-size 128 \
    --epochs 20 \
    --opt adamw \
    --lr 0.0001 \
    --weight-decay 0.05 \
    --sched cosine \
    --warmup-lr 0.00001 \
    --min-lr 0.000001 \
    --warmup-epochs 2 \
    --seed 42 \
    --gpu-id 0 \
    --workers 4 \
    --output ./outputs/teacher

