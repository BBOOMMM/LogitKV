CUDA_VISIBLE_DEVICES=0 python evaluate.py \
  --dataset ruler \
  --data_dir "$KVPRESS_DATASETS/ruler/32768" \
  --model "$MODELS_DIR/Meta-Llama-3.1-8B-Instruct" \
  --press_name logitkv \
  --compression_ratio 0.6 \
  --fisher_window 32 \
  --fisher_positions 2 \
  --fisher_labels 2 \
  --fisher_position_aggregation mean \
  --score_mode coupled_diag \
  --coupled_kernel_size 1 \
  --coupled_pooling avg \
  --first_stage_ratio 0.5 \
  --fisher_seed 42 \
  --attention_eps 0 \
  --device cuda:0 \
  --fraction 0.2 \
  --snapkv_kernel_size 7 \
  --tasks niah_multikey_2,niah_multikey_3


# cwe
# fwe
# niah_multikey_1
# niah_multikey_2
# niah_multikey_3
# niah_multiquery
# niah_multivalue
# niah_single_1
# niah_single_2
# niah_single_3
# qa_1
# qa_2
# vt


# 先在明显差的上面跑一下 0.2 看看是不是样本差异