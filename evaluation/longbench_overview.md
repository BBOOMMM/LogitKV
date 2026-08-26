# LongBench Context 统计

当前预处理数据包含 **16 个子任务、3,315 个唯一 context**。

## 按类别汇总

| 类别 | 子任务数 | Context 总数 |
|:---|---:|---:|
| SingleDoc QA | 3 | 280 |
| multidoc QA | 3 | 600 |
| summarization | 3 | 435 |
| fewshot | 3 | 600 |
| synthetic | 2 | 400 |
| code | 2 | 1,000 |
| **合计** | **16** | **3,315** |

## 子任务明细

| 类别 | 子任务 | Context 数 |
|:---|:---|---:|
| SingleDoc QA | `narrativeqa` | 20 |
| SingleDoc QA | `qasper` | 148 |
| SingleDoc QA | `multifieldqa_en` | 112 |
| multidoc QA | `hotpotqa` | 200 |
| multidoc QA | `2wikimqa` | 200 |
| multidoc QA | `musique` | 200 |
| summarization | `gov_report` | 200 |
| summarization | `qmsum` | 35 |
| summarization | `multi_news` | 200 |
| fewshot | `trec` | 200 |
| fewshot | `triviaqa` | 200 |
| fewshot | `samsum` | 200 |
| synthetic | `passage_count` | 200 |
| synthetic | `passage_retrieval_en` | 200 |
| code | `lcc` | 500 |
| code | `repobench-p` | 500 |
| **合计** | **16 个子任务** | **3,315** |

> 注：这里的 Context 数是每个任务中唯一 `context` 的数量，不是数据行数。一个 context 可能对应多个问题。
