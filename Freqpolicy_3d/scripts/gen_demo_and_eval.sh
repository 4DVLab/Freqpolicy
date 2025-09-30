#!/bin/bash
tasks=(
    "disassemble"
    "box-close"
    "bin-picking"
    "coffee-pull"
    "coffee-push"
    "drawer-open"
    "hammer"
    "hand-insert"
    "handle-press"
    "lever-pull"
    "peg-insert-side"
    "peg-unplug-side"
    "pick-out-of-hole"
    "pick-place-wall"
    "pick-place"
    "push-wall"
    "push"
    "soccer"
    "shelf-place"
    "stick-pull"
    "sweep"
)

# 设置通用参数
MODEL="dp3"
ADDITION_INFO="0428_new1"
GPU_ID="0"
SEED="0"

# 设置CUDA设备
export CUDA_VISIBLE_DEVICES=${GPU_ID}

# 为所有任务循环执行
for TASK in "${tasks[@]}"
do
    TASK_FULL="metaworld_${TASK}"
    
    echo "=================================================================="
    echo "开始处理任务: ${TASK}"
    echo "=================================================================="
    
    # 检查专家演示数据是否已存在
    DEMO_PATH="3D-Diffusion-Policy/data/metaworld_${TASK}_expert.zarr/data"
    
    if [ -d "$DEMO_PATH" ]; then
        echo "专家演示数据已存在，跳过生成步骤..."
    else
        echo "===== 第1步：生成专家演示 ====="
        
        # 进入Metaworld目录生成专家演示
        cd third_party/Metaworld
        
        # 生成专家演示
        python gen_demonstration_expert.py --env_name=${TASK} \
            --num_episodes 20 \
            --root_dir "../../3D-Diffusion-Policy/data/"
        
        # 返回上级目录
        cd ../..
        
        echo "任务 ${TASK} 的专家演示数据生成完成"
    fi
    
    echo "===== 第2步：执行模型评估 ====="
    
    # 执行评估脚本
    bash scripts/eval_policy.sh ${MODEL} ${TASK_FULL} ${ADDITION_INFO} ${SEED} ${GPU_ID}
    
    echo "任务 ${TASK} 的模型评估完成"
    
    # 短暂休息
    sleep 2
done

echo "=================================================================="
echo "所有任务处理完成！"
echo "==================================================================" 
