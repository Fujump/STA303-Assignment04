#!/bin/bash

# 定义要运行的命令的基本部分
BASE_COMMAND="python conformal_training_0.py"

# 定义要重复的参数组合   "CIFAR10"
WEIGHTS=(0.05 0.1 0.2 0.3 0.4)

PENALTIES=(0.001 0.01 0.1)

# MODEL_NAMES=("resnet18" "resnet34" "resnet50" "resnet101" "resnet152" "vgg11")

# DATASETS=("cifar100" "cifar10")

SCORES=("SAPS" "RAPS" "APS" "THR")

PREDICTORS=("ClusterPredictor" "ClassWisePredictor" "SplitPredictor")

# 循环嵌套，运行所有组合
for score in "${SCORES[@]}"; do
    for predictor in "${PREDICTORS[@]}"; do
        # 构建完整的命令并运行
        FULL_COMMAND="$BASE_COMMAND --score $score --predictor $predictor"
        echo "Running command: $FULL_COMMAND"
        $FULL_COMMAND
    done
done