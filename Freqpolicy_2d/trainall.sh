#!/bin/bash

# Test script for all tasks - runs each task for 2 minutes
# Usage: ./test_all_tasks.sh

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Arrays to store results
declare -a SUCCESSFUL_TASKS=()
declare -a FAILED_TASKS=()
declare -a ERROR_MESSAGES=()

# Function to run a single task
run_task() {
    local config_dir=$1
    local config_name=$2
    local task_display_name="${config_dir}/${config_name}"
    
    echo -e "${YELLOW}[$(date '+%H:%M:%S')] Testing: ${task_display_name}${NC}"
    
    # Create log file for this task
    local log_file="test_logs/${config_dir}_${config_name%.yaml}.log"
    mkdir -p test_logs
    
    # Run the task with 2 minute timeout
    timeout 1200s python train.py \
        --config-dir=config_task/${config_dir} \
        --config-name=${config_name} \
        training.seed=42 \
        training.device=cuda:0 \
        hydra.run.dir="data/test_outputs/\${now:%Y.%m.%d}/\${now:%H.%M.%S}_\${name}_\${task_name}" \
        > "${log_file}" 2>&1
    
    local exit_code=$?
    
    # Check if task failed (exit code != 0 and != 124)
    # 124 is timeout exit code, which is expected
    if [ $exit_code -eq 124 ]; then
        echo -e "${GREEN}[$(date '+%H:%M:%S')] ✓ ${task_display_name} - Timeout reached (2min) - OK${NC}"
        SUCCESSFUL_TASKS+=("${task_display_name}")
    elif [ $exit_code -eq 0 ]; then
        echo -e "${GREEN}[$(date '+%H:%M:%S')] ✓ ${task_display_name} - Completed successfully${NC}"
        SUCCESSFUL_TASKS+=("${task_display_name}")
    else
        echo -e "${RED}[$(date '+%H:%M:%S')] ✗ ${task_display_name} - FAILED (exit code: $exit_code)${NC}"
        FAILED_TASKS+=("${task_display_name}")
        
        # Extract error message from log
        local error_msg=$(tail -10 "${log_file}" | grep -E "(Error|Exception|Failed)" | head -1)
        if [ -z "$error_msg" ]; then
            error_msg="Check log file: ${log_file}"
        fi
        ERROR_MESSAGES+=("${task_display_name}: ${error_msg}")
    fi
    
    echo "---"
}

# Main execution
echo -e "${YELLOW}Starting task validation test...${NC}"
echo "Each task will run for maximum 2 minutes"
echo "=================================="

# Test image tasks
echo -e "${YELLOW}Testing IMAGE tasks:${NC}"
image_tasks=(
    "can_mh.yaml"
    "can_ph.yaml" 
    "lift_mh.yaml"
    "lift_ph.yaml"
    "pusht.yaml"
    "square_mh.yaml"
    "square_ph.yaml"
    "tool_hang_ph.yaml"
    "transport_mh.yaml"
    "transport_ph.yaml"
)

for task in "${image_tasks[@]}"; do
    run_task "image" "$task"
done

# # Test low_dim tasks
echo -e "${YELLOW}Testing LOW_DIM tasks:${NC}"
low_dim_tasks=(
    "can_mh.yaml"
    "can_ph.yaml"
    "lift_mh.yaml" 
    "lift_ph.yaml"
    "pusht.yaml"
    "square_mh.yaml"
    "square_ph.yaml"
    "tool_hang_ph.yaml"
    "transport_mh.yaml"
    "transport_ph.yaml"
)

for task in "${low_dim_tasks[@]}"; do
    run_task "low_dim" "$task"
done

# Generate final report
echo "=================================="
echo -e "${YELLOW}FINAL REPORT:${NC}"
echo "=================================="

echo -e "${GREEN}SUCCESSFUL TASKS (${#SUCCESSFUL_TASKS[@]}):${NC}"
for task in "${SUCCESSFUL_TASKS[@]}"; do
    echo -e "  ✓ ${task}"
done

if [ ${#FAILED_TASKS[@]} -gt 0 ]; then
    echo ""
    echo -e "${RED}FAILED TASKS (${#FAILED_TASKS[@]}):${NC}"
    for i in "${!FAILED_TASKS[@]}"; do
        echo -e "  ✗ ${FAILED_TASKS[$i]}"
        echo -e "    ${ERROR_MESSAGES[$i]}"
    done
    
    echo ""
    echo -e "${RED}需要检查的文件:${NC}"
    for task in "${FAILED_TASKS[@]}"; do
        echo -e "  - config_task/${task}"
    done
else
    echo -e "${GREEN}🎉 All tasks passed initial validation!${NC}"
fi

echo ""
echo "详细日志文件保存在: test_logs/ 文件夹"
echo "测试完成时间: $(date)"

# Exit with error code if any tasks failed
if [ ${#FAILED_TASKS[@]} -gt 0 ]; then
    exit 1
else
    exit 0
fi