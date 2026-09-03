#!/usr/bin/env bash

set -euo pipefail

# LongBench contains a few contexts close to the model limit.  Expandable
# segments reduce allocator fragmentation across the many independent
# context-level prefill/backward runs.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# The defaults are the current LongBench-tuned LogitKV configuration.  For
# quick experiments, set LONGBENCH_TASKS (comma-separated task names/groups)
# and/or LONGBENCH_FRACTION before invoking this script.
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
eval_args=(
  --dataset longbench
  --data_dir "${KVPRESS_DATASETS}/longbench"
  --model "${MODELS_DIR}/Meta-Llama-3.1-8B-Instruct"
  --press_name "${LONGBENCH_PRESS_NAME:-logit_snapkv}"
  --fisher_seed "${LONGBENCH_FISHER_SEED:-42}"
  --attention_eps "${LONGBENCH_ATTENTION_EPS:-0}"
  --fisher_eps "${LONGBENCH_FISHER_EPS:-1e-9}"
  --device "${LONGBENCH_DEVICE:-cuda:0}"
  --compression_ratio "${LONGBENCH_COMPRESSION_RATIO:-0.8}"
  --snapkv_window_size "${LONGBENCH_SNAPKV_WINDOW:-32}"
  --fisher_window "${LONGBENCH_FISHER_WINDOW:-32}"
  --fisher_positions "${LONGBENCH_FISHER_POSITIONS:-1}"
  --fisher_labels "${LONGBENCH_FISHER_LABELS:-4}"
  --fisherlabel_samplemode "${LONGBENCH_FISHER_LABEL_SAMPLE_MODE:-multinomial}"
  --fisher_backward_mode "${LONGBENCH_FISHER_BACKWARD_MODE:-label_sketch}"
  --fisher_sketches "${LONGBENCH_FISHER_SKETCHES:-4}"
  --fisher_position_aggregation "${LONGBENCH_FISHER_POSITION_AGGREGATION:-mean}"
  --score_mode "${LONGBENCH_SCORE_MODE:-coupled_diag}"
  --coupled_kernel_size "${LONGBENCH_COUPLED_KERNEL:-1}"
  --coupled_pooling "${LONGBENCH_COUPLED_POOLING:-avg}"
  --coupled_effect_mode "${LONGBENCH_COUPLED_EFFECT_MODE:-value}"
  --coupled_attention_power "${LONGBENCH_COUPLED_ATTENTION_POWER:-0}"
  --coupled_key_weight "${LONGBENCH_COUPLED_KEY_WEIGHT:-0}"
  --fisher_value_l2_weight "${LONGBENCH_FISHER_VALUE_L2_WEIGHT:-0}"
  --fisher_probe_normalization "${LONGBENCH_FISHER_PROBE_NORMALIZATION:-none}"
  --first_stage_ratio "${LONGBENCH_FIRST_STAGE_RATIO:-0.5}"
  --allocation_mode "${LONGBENCH_ALLOCATION_MODE:-adaptive}"
  --alpha_safeguard "${LONGBENCH_ALPHA_SAFEGUARD:-0.2}"
  --fraction "${LONGBENCH_FRACTION:-1}"
  --snapkv_kernel_size "${LONGBENCH_SNAPKV_KERNEL:-5}"
)

if [[ -n "${LONGBENCH_TASKS:-}" ]]; then
  eval_args+=(--tasks "${LONGBENCH_TASKS}")
fi
if [[ -n "${LONGBENCH_MAX_NEW_TOKENS:-}" ]]; then
  eval_args+=(--max_new_tokens "${LONGBENCH_MAX_NEW_TOKENS}")
fi
if [[ -n "${LONGBENCH_MAX_CONTEXT_LENGTH:-}" ]]; then
  eval_args+=(--max_context_length "${LONGBENCH_MAX_CONTEXT_LENGTH}")
fi
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python "${script_dir}/evaluate.py" "${eval_args[@]}"
