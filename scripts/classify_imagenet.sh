#!/bin/bash

# 定义要运行的命令的基本部分
BASE_COMMAND="python imagenet_example.py"

# 定义要重复的参数组合 "resnet18" "resnet34" "resnet50" "resnet101" "resnet152"  "vgg13" "vgg16" "vgg19"
WEIGHTS=(0.05 0.1 0.2 0.3 0.4)

PENALTIES=(0.001 0.01 0.1)

MODEL_NAMES=("vgg11")
#  "RAPS" "THR" "APS" 
SCORES=("SAPS")
#  "ClusterPredictor" "ClassWisePredictor"
PREDICTORS=("SplitPredictor")

# 循环嵌套，运行所有组合
for weight in "${WEIGHTS[@]}"; do
    for model_name in "${MODEL_NAMES[@]}"; do
        for predictor in "${PREDICTORS[@]}"; do
            # 构建完整的命令并运行
            FULL_COMMAND="$BASE_COMMAND --model_name $model_name --weight $weight --score SAPS --predictor $predictor"
            echo "Running command: $FULL_COMMAND"
            $FULL_COMMAND
        done
    done
done

for penalty in "${PENALTIES[@]}"; do
    for model_name in "${MODEL_NAMES[@]}"; do
        for predictor in "${PREDICTORS[@]}"; do
            # 构建完整的命令并运行
            FULL_COMMAND="$BASE_COMMAND --model_name $model_name --penalty $penalty --score RAPS --predictor $predictor"
            echo "Running command: $FULL_COMMAND"
            $FULL_COMMAND
        done
    done
done

