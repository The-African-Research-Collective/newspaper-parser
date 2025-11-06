#!/usr/bin/env bash
# ============================================================
# Script: train_karanta_ocr.sh
# Description: Launch OCR fine-tuning with Accelerate on multiple GPUs
#              Runs in background and logs output to specified file
# ============================================================

# ---- LOG FILE ARGUMENT ----
LOG_FILE=$1
if [[ -z "$LOG_FILE" ]]; then
    echo "❌ Error: No log file specified."
    echo "Usage: $0 <log_file_path>"
    exit 1
fi

# ---- CONFIGURATION ----
export NUM_MACHINES=1
export NUM_PROCESSES=8
export MAIN_PORT=29501
export MIXED_PRECISION=bf16
export CONFIG_PATH=$2

# ---- OPTIONAL ENV VARS ----
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=false
export HF_HOME="$HOME/.cache/huggingface"
export HF_DATASETS_CACHE="$HOME/.cache/huggingface/datasets"
export WANDB_PROJECT="karanta-ocr"
export NCCL_P2P_DISABLE=1
export NCCL_DEBUG=INFO
export PYTHONUNBUFFERED=1
export WANDB_WATCH=all
export NCCL_TIMEOUT=5400 # prevent NCCL timeouts among processes
# ---- PATHS ----
SCRIPT_PATH="karanta/training/ocr_training.py"

# Ensure directory for log file exists
mkdir -p "$(dirname "$LOG_FILE")"

# ---- DEVICE CHECK ----
IFS=',' read -ra GPU_ARRAY <<< "$CUDA_VISIBLE_DEVICES"
NUM_GPUS=${#GPU_ARRAY[@]}

if (( NUM_PROCESSES > NUM_GPUS )); then
    echo "❌ Error: NUM_PROCESSES ($NUM_PROCESSES) exceeds available GPUs ($NUM_GPUS) from CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    echo "Please reduce NUM_PROCESSES or adjust CUDA_VISIBLE_DEVICES."
    exit 1
fi

# ---- LAUNCH ----
echo "✅ Launching Accelerate training..."
echo "   NUM_MACHINES = $NUM_MACHINES"
echo "   NUM_PROCESSES = $NUM_PROCESSES"
echo "   CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES"
echo "   MIXED_PRECISION = $MIXED_PRECISION"
echo "   CONFIG_PATH = $CONFIG_PATH"
echo "   LOG_FILE = $LOG_FILE"
echo

nohup accelerate launch \
  --mixed_precision "$MIXED_PRECISION" \
  --num_machines "$NUM_MACHINES" \
  --num_processes "$NUM_PROCESSES" \
  --main_process_port "$MAIN_PORT" \
  "$SCRIPT_PATH" "$CONFIG_PATH" \
  > "$LOG_FILE" 2>&1 &

PID=$!
echo "🚀 Training started in background (PID: $PID)"
echo "📜 Logs are being written to: $LOG_FILE"
echo "Use 'tail -f $LOG_FILE' to monitor progress."
