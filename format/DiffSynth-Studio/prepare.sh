#!/usr/bin/env bash
# Run inside an installed DiffSynth-Studio environment.
set -euo pipefail
: "${DIFFSYNTH_REPO:?Set DIFFSYNTH_REPO to the upstream checkout}"
sample_root="$(cd "$(dirname "$0")" && pwd)"
cd "$DIFFSYNTH_REPO"
accelerate launch examples/minimax_h3/model_training/train.py \
  --dataset_base_path "$sample_root" \
  --dataset_metadata_path "$sample_root/metadata.json" \
  --data_file_keys "video,input_audio,references" \
  --extra_inputs "input_audio,references" \
  --height 480 --width 832 --num_frames 124 --dataset_repeat 1 \
  --model_id_with_origin_paths "MiniMax/MiniMax-H3:Ref2VA/text_encoder/model*.safetensors,MiniMax/MiniMax-H3:Ref2VA/video_vae/source/model.safetensors,MiniMax/MiniMax-H3:Ref2VA/audio_vae/model.safetensors" \
  --processor_path "MiniMax/MiniMax-H3:Ref2VA/processor/" \
  --learning_rate 1e-4 --num_epochs 1 --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "$sample_root/cache" \
  --lora_base_model dit --lora_target_modules "attn.qkv_proj,attn.out_proj,mlp.fc1,mlp.fc2" \
  --lora_rank 32 --use_gradient_checkpointing --task "sft:data_process"
