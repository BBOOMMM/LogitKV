# LogitKV LongBench 配置与复现记录

本文档记录当前用于比较的 LogitKV LongBench 全量结果。配置按任务分别选择；最终宏平均是 16 个任务分数的平均值。所有评测均通过 `evaluation/run_longbench.sh` 执行，没有复用 CriticalKV 的结果或中间状态。

## 评测环境与公共参数

- 数据集：`longbench`
- 模型：`$MODELS_DIR/Meta-Llama-3.1-8B-Instruct`
- 压缩率：`0.8`
- SnapKV window/kernel：`32 / 5`
- Fisher window：`32`
- Fisher seed：`42`
- Fisher backward：`label_sketch`
- Fisher sketches：`4`
- allocation：`adaptive`
- `first_stage_ratio`：`0.5`，除非任务表另有说明
- `attention_eps=0`，`fisher_eps=1e-9`
- `coupled_kernel_size=1`，`coupled_pooling=avg`
- `coupled_effect_mode=value`，`coupled_key_weight=0`
- `fraction=1`

脚本从 `$MODELS_DIR` 和 `$KVPRESS_DATASETS` 读取模型及数据集路径。下面的命令假定当前目录是仓库根目录，并且这两个环境变量已经设置。

## 各任务最终配置

未列出的参数使用上面的公共值或 `run_longbench.sh` 中的默认值。

| 任务 | score mode | positions / labels | label sample / position aggregation | attention power | first stage | alpha safeguard | 全量分数 |
|---|---|---:|---|---:|---:|---:|---:|
| `2wikimqa` | `coupled_diag` | `1 / 4` | `multinomial / mean` | `0` | `0.5` | `0` | 33.43 |
| `gov_report` | `coupled_diag` | `1 / 4` | `multinomial / mean` | `0` | `0.5` | `0.2` | 27.63 |
| `hotpotqa` | `coupled_diag` | `1 / 4` | `multinomial / mean` | `0` | `0.5` | `0.2` | 50.12 |
| `lcc` | `separable` | `1 / 4` | `multinomial / mean` | — | `0.5` | `0.2` | 68.12 |
| `multi_news` | `separable` | `1 / 4` | `multinomial / mean` | — | `0` | `0.2` | 22.75 |
| `multifieldqa_en` | `coupled_diag` | `16 / 1` | `top_fisherposition / max` | `0` | `0.5` | `0.2` | 37.24 |
| `musique` | `coupled_diag` | `1 / 4` | `multinomial / mean` | `0.25` | `0.5` | `0.2` | 24.90 |
| `narrativeqa` | `coupled_diag` | `16 / 1` | `top_fisherposition / max` | `0` | `0.5` | `0.2` | 29.27 |
| `passage_count` | `coupled_diag` | `1 / 4` | `multinomial / mean` | `0` | `0.5` | `0.2` | 8.56 |
| `passage_retrieval_en` | `coupled_diag` | `1 / 4` | `multinomial / mean` | `0` | `0.5` | `0.2` | 99.00 |
| `qasper` | `coupled_diag` | `8 / 1` | `top_fisherposition / max` | `0.1` | `0.5` | `0.2` | 32.02 |
| `qmsum` | `separable` | `1 / 4` | `multinomial / mean` | — | `0` | `0.2` | 21.99 |
| `repobench-p` | `separable` | `1 / 4` | `multinomial / mean` | — | `0.5` | `0.2` | 56.63 |
| `samsum` | `coupled_diag` | `1 / 4` | `multinomial / mean` | `0` | `1` | `0.2` | 43.77 |
| `trec` | `separable` | `1 / 4` | `multinomial / mean` | — | `0.5` | `0.2` | 54.00 |
| `triviaqa` | `coupled_diag` | `1 / 4` | `multinomial / mean` | `0` | `1` | `0.2` | 92.38 |

## 可复制的运行命令

先设置公共环境变量：

```bash
export LONGBENCH_FRACTION=1
export LONGBENCH_FISHER_SEED=42
export LONGBENCH_SNAPKV_WINDOW=32
export LONGBENCH_SNAPKV_KERNEL=5
export LONGBENCH_FISHER_WINDOW=32
export LONGBENCH_FISHER_BACKWARD_MODE=label_sketch
export LONGBENCH_FISHER_SKETCHES=4
export LONGBENCH_ALLOCATION_MODE=adaptive
export LONGBENCH_FIRST_STAGE_RATIO=0.5
export LONGBENCH_ALPHA_SAFEGUARD=0.2
export LONGBENCH_ATTENTION_EPS=0
export LONGBENCH_FISHER_EPS=1e-9
```

逐任务执行以下命令即可复现当前配置。连续执行时，注意每条命令都显式覆盖了前一任务的特殊参数：

```bash
# 2wikimqa
LONGBENCH_TASKS=2wikimqa \
LONGBENCH_SCORE_MODE=coupled_diag LONGBENCH_FISHER_POSITIONS=1 LONGBENCH_FISHER_LABELS=4 \
LONGBENCH_FISHER_LABEL_SAMPLE_MODE=multinomial LONGBENCH_FISHER_POSITION_AGGREGATION=mean \
LONGBENCH_COUPLED_ATTENTION_POWER=0 LONGBENCH_ALPHA_SAFEGUARD=0 \
bash evaluation/run_longbench.sh

# gov_report
LONGBENCH_TASKS=gov_report \
LONGBENCH_SCORE_MODE=coupled_diag LONGBENCH_FISHER_POSITIONS=1 LONGBENCH_FISHER_LABELS=4 \
LONGBENCH_FISHER_LABEL_SAMPLE_MODE=multinomial LONGBENCH_FISHER_POSITION_AGGREGATION=mean \
LONGBENCH_COUPLED_ATTENTION_POWER=0 LONGBENCH_FIRST_STAGE_RATIO=0.5 \
bash evaluation/run_longbench.sh

# hotpotqa
LONGBENCH_TASKS=hotpotqa \
LONGBENCH_SCORE_MODE=coupled_diag LONGBENCH_FISHER_POSITIONS=1 LONGBENCH_FISHER_LABELS=4 \
LONGBENCH_FISHER_LABEL_SAMPLE_MODE=multinomial LONGBENCH_FISHER_POSITION_AGGREGATION=mean \
LONGBENCH_COUPLED_ATTENTION_POWER=0 \
bash evaluation/run_longbench.sh

# lcc
LONGBENCH_TASKS=lcc LONGBENCH_SCORE_MODE=separable \
LONGBENCH_FISHER_POSITIONS=1 LONGBENCH_FISHER_LABELS=4 \
LONGBENCH_FISHER_LABEL_SAMPLE_MODE=multinomial LONGBENCH_FISHER_POSITION_AGGREGATION=mean \
LONGBENCH_FIRST_STAGE_RATIO=0.5 \
bash evaluation/run_longbench.sh

# multi_news, qmsum：两者均为 separable 且 first_stage_ratio=0
for task in multi_news qmsum; do
  LONGBENCH_TASKS="$task" LONGBENCH_SCORE_MODE=separable \
  LONGBENCH_FISHER_POSITIONS=1 LONGBENCH_FISHER_LABELS=4 \
  LONGBENCH_FISHER_LABEL_SAMPLE_MODE=multinomial LONGBENCH_FISHER_POSITION_AGGREGATION=mean \
  LONGBENCH_FIRST_STAGE_RATIO=0 bash evaluation/run_longbench.sh
done

# multifieldqa_en
LONGBENCH_TASKS=multifieldqa_en \
LONGBENCH_SCORE_MODE=coupled_diag LONGBENCH_FISHER_POSITIONS=16 LONGBENCH_FISHER_LABELS=1 \
LONGBENCH_FISHER_LABEL_SAMPLE_MODE=top_fisherposition LONGBENCH_FISHER_POSITION_AGGREGATION=max \
LONGBENCH_COUPLED_ATTENTION_POWER=0 \
bash evaluation/run_longbench.sh

# musique
LONGBENCH_TASKS=musique \
LONGBENCH_SCORE_MODE=coupled_diag LONGBENCH_FISHER_POSITIONS=1 LONGBENCH_FISHER_LABELS=4 \
LONGBENCH_FISHER_LABEL_SAMPLE_MODE=multinomial LONGBENCH_FISHER_POSITION_AGGREGATION=mean \
LONGBENCH_COUPLED_ATTENTION_POWER=0.25 \
bash evaluation/run_longbench.sh

# narrativeqa
LONGBENCH_TASKS=narrativeqa \
LONGBENCH_SCORE_MODE=coupled_diag LONGBENCH_FISHER_POSITIONS=16 LONGBENCH_FISHER_LABELS=1 \
LONGBENCH_FISHER_LABEL_SAMPLE_MODE=top_fisherposition LONGBENCH_FISHER_POSITION_AGGREGATION=max \
LONGBENCH_COUPLED_ATTENTION_POWER=0 \
bash evaluation/run_longbench.sh

# passage_count, passage_retrieval_en：共享同一配置
for task in passage_count passage_retrieval_en; do
  LONGBENCH_TASKS="$task" LONGBENCH_SCORE_MODE=coupled_diag \
  LONGBENCH_FISHER_POSITIONS=1 LONGBENCH_FISHER_LABELS=4 \
  LONGBENCH_FISHER_LABEL_SAMPLE_MODE=multinomial LONGBENCH_FISHER_POSITION_AGGREGATION=mean \
  LONGBENCH_COUPLED_ATTENTION_POWER=0 bash evaluation/run_longbench.sh
done

# qasper
LONGBENCH_TASKS=qasper \
LONGBENCH_SCORE_MODE=coupled_diag LONGBENCH_FISHER_POSITIONS=8 LONGBENCH_FISHER_LABELS=1 \
LONGBENCH_FISHER_LABEL_SAMPLE_MODE=top_fisherposition LONGBENCH_FISHER_POSITION_AGGREGATION=max \
LONGBENCH_COUPLED_ATTENTION_POWER=0.1 \
bash evaluation/run_longbench.sh

# repobench-p, trec：separable 配置
for task in repobench-p trec; do
  LONGBENCH_TASKS="$task" LONGBENCH_SCORE_MODE=separable \
  LONGBENCH_FISHER_POSITIONS=1 LONGBENCH_FISHER_LABELS=4 \
  LONGBENCH_FISHER_LABEL_SAMPLE_MODE=multinomial LONGBENCH_FISHER_POSITION_AGGREGATION=mean \
  LONGBENCH_FIRST_STAGE_RATIO=0.5 bash evaluation/run_longbench.sh
done

# samsum, triviaqa：coupled 配置且 first_stage_ratio=1
for task in samsum triviaqa; do
  LONGBENCH_TASKS="$task" LONGBENCH_SCORE_MODE=coupled_diag \
  LONGBENCH_FISHER_POSITIONS=1 LONGBENCH_FISHER_LABELS=4 \
  LONGBENCH_FISHER_LABEL_SAMPLE_MODE=multinomial LONGBENCH_FISHER_POSITION_AGGREGATION=mean \
  LONGBENCH_COUPLED_ATTENTION_POWER=0 LONGBENCH_FIRST_STAGE_RATIO=1 \
  bash evaluation/run_longbench.sh
done
```

## 结果汇总

参考目标文件：`evaluation/results/longbench/longbench__Meta-Llama-3.1-8B-Instruct__criti_snapkv__cr0.8__sk5__frac1.00.json`。

| 类别 | LogitKV | critic_snapkv |
|---|---:|---:|
| SingleDoc QA | 32.8433 | 31.7367 |
| Multidoc QA | 36.1500 | 37.4267 |
| Summarization | 24.1233 | 24.7233 |
| Few-shot | 63.3833 | 63.2933 |
| Synthetic | 53.7800 | 52.4100 |
| Code | 62.3750 | 62.6300 |
| **16-task macro average** | **43.863125** | **43.851250** |

当前宏平均领先 `0.011875`。每个任务的 JSON/CSV 结果保存在 `evaluation/results/longbench/`，文件名包含完整超参数标签；例如 qasper 的最终结果是：

`longbench__Meta-Llama-3.1-8B-Instruct__logit_snapkv__cr0.8__sk5__sw32__fw32__positions8__labels1__modecoupled_diag__fbmlabel_sketch__fsk4__paggmax__ck1__cap0.1__allocadaptive__alpha0.2__sr0.5__fs42__ae0__flsmtop__frac1.00__tasksqasper.json`
