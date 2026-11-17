#!/bin/bash

total_time=10
agent_num=6

declare -a epoch_choice=("1" "10" "20" "40" "60" "80" "100")
declare -a types=("mnist" "cifar10")
declare -a methods=("naive" "lwf" "ewc" "gem")
declare -a backbone_choices=("0" "1")

for num_epochs in "${epoch_choice[@]}"; do
    for dataset_type in "${types[@]}"; do
        for method in "${methods[@]}"; do
            for is_backbone in "${backbone_choices[@]}"; do
                python lifelong_baselines/check_lifelong.py \
                -num_epochs=$num_epochs \
                -dataset_type=$dataset_type \
                -method=$method \
                -agent_num=$agent_num \
                -total_time=$total_time \
                -is_backbone=$is_backbone
            done
        done
    done
done

wait

echo "All processes completed!"
