# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import logging
from pathlib import Path
import re
import time
from typing import Optional

import torch
from datasets import load_dataset, load_from_disk
import pandas as pd
from fire import Fire
import transformers
from infinite_bench.calculate_metrics import calculate_metrics as infinite_bench_scorer
from longbench.calculate_metrics import calculate_metrics as longbench_scorer
from kvpress.ada_attn import replace_var_flash_attn
from loogle.calculate_metrics import calculate_metrics as loogle_scorer
from ruler.calculate_metrics import calculate_metrics as ruler_scorer
from tqdm import tqdm
from transformers import pipeline
from zero_scrolls.calculate_metrics import calculate_metrics as zero_scrolls_scorer
import warnings

warnings.filterwarnings(
    "ignore", message="MatMul8bitLt: inputs will be cast from torch.bfloat16 to float16 during quantization"
)

from kvpress import (
    AdaKVPress,
    ChunkKVPress,
    CriticalKVPress,
    CriticalAdaKVPress,
    CriticalKVPress,
    DuoAttentionPress,
    ExpectedAttentionPress,
    KnormPress,
    LogitKVPress,
    ObservedAttentionPress,
    RandomPress,
    SnapKVPress,
    StreamingLLMPress,
    ThinKPress,
    TOVAPress,
    EfficientDefensiveKVPress,
    EfficientLayerDefensiveKVPress,
    EfficientAdaSnapKVPress,
    EfficientAdaScorerPress,
    EfficientAdaGlobalScorerPress,
    ThinKPress,
    TOVAPress,
    CakeGlobalPress,
    # CakeScorerPress,
)
from kvpress.presses.logitkv_press import FISHER_LABEL_SAMPLE_MODES

logger = logging.getLogger(__name__)

DATASET_DICT = {
    "loogle": "simonjegou/loogle",
    "ruler": "simonjegou/ruler",
    "zero_scrolls": "simonjegou/zero_scrolls",
    "infinitebench": "MaxJeblick/InfiniteBench",
    "longbench": None,
}

SCORER_DICT = {
    "loogle": loogle_scorer,
    "ruler": ruler_scorer,
    "zero_scrolls": zero_scrolls_scorer,
    "infinitebench": infinite_bench_scorer,
    "longbench": longbench_scorer,
}

PRESS_DICT = {
    # Keep the legacy name for callers that still use it. The two explicit
    # names below use independent press instances and are the recommended
    # configurations for LogitKV-vs-CriticalKV comparisons.
    "logitkv": LogitKVPress(SnapKVPress(), fisher_seed=42),
    "logit_snapkv": LogitKVPress(SnapKVPress(), fisher_seed=42),
    # Backward-compatible alias for the historical typo.
    "logit_sanpkv": LogitKVPress(SnapKVPress(), fisher_seed=42),
    "logit_adasnapkv": LogitKVPress(
        SnapKVPress(), allocation_mode="adaptive", fisher_seed=42
    ),
    "criti_adasnapkv": CriticalAdaKVPress(SnapKVPress()),
    "criti_ada_expected_attention": CriticalAdaKVPress(ExpectedAttentionPress(use_vnorm=False)),
    "criti_snapkv": CriticalKVPress(SnapKVPress()),
    "criti_expected_attention": CriticalKVPress(ExpectedAttentionPress(use_vnorm=False)),
    "adasnapkv": AdaKVPress(SnapKVPress()),
    "ada_expected_attention": AdaKVPress(ExpectedAttentionPress()),
    "expected_attention": ExpectedAttentionPress(),
    "ada_expected_attention_e2": AdaKVPress(ExpectedAttentionPress(epsilon=1e-2)),
    "knorm": KnormPress(),
    "observed_attention": ObservedAttentionPress(),
    "random": RandomPress(),
    "snapkv": SnapKVPress(),
    "streaming_llm": StreamingLLMPress(),
    "think": ThinKPress(),
    "tova": TOVAPress(),
    "duo_attention": DuoAttentionPress(),
    "chunkkv": ChunkKVPress(press=SnapKVPress(), chunk_length=20),
    "efficient_ada_snapkv": EfficientAdaSnapKVPress(),
    "efficient_defensivekv": EfficientDefensiveKVPress(),
    "efficient_layer_defensivekv": EfficientLayerDefensiveKVPress(),
    "cake_global": CakeGlobalPress(),
}


def _configure_snapkv_kernel_size(press, kernel_size: int) -> bool:
    """Set the pooling kernel on a SnapKV-based press, if it has one."""

    current = press
    while current is not None:
        if isinstance(current, (SnapKVPress, EfficientAdaSnapKVPress)):
            current.kernel_size = kernel_size
            return True
        current = getattr(current, "press", None)
    return False


LONGBENCH_TASK_GROUPS = {
    "single_doc_qa": ["narrativeqa", "qasper", "multifieldqa_en"],
    "multidoc_qa": ["hotpotqa", "2wikimqa", "musique"],
    "summarization": ["gov_report", "qmsum", "multi_news"],
    "fewshot": ["trec", "triviaqa", "samsum"],
    "synthetic": ["passage_count", "passage_retrieval_en"],
    "code": ["lcc", "repobench-p"],
}

LONGBENCH_TASK_GROUP_LABELS = {
    "single_doc_qa": "SingleDoc QA",
    "multidoc_qa": "multidoc QA",
    "summarization": "summarization",
    "fewshot": "fewshot",
    "synthetic": "synthetic",
    "code": "code",
}

# Also accept the labels used in the LongBench task listing, e.g. "SingleDoc
# QA" and "multidoc QA", after normalizing spaces and punctuation.
LONGBENCH_TASK_GROUP_ALIASES = {
    "single_doc_qa": "single_doc_qa",
    "singledoc_qa": "single_doc_qa",
    "single_doc": "single_doc_qa",
    "singledoc": "single_doc_qa",
    "multidoc_qa": "multidoc_qa",
    "multi_doc_qa": "multidoc_qa",
    "multidoc": "multidoc_qa",
    "multi_doc": "multidoc_qa",
    "summarization": "summarization",
    "summary": "summarization",
    "sum": "summarization",
    "fewshot": "fewshot",
    "few_shot": "fewshot",
    "synthetic": "synthetic",
    "code": "code",
}


def _normalize_task_selector(selector: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", selector.strip().lower()).strip("_")


def _resolve_requested_tasks(dataset: str, tasks=None, task=None):
    """Expand task-group selectors and preserve support for exact task names."""
    selectors = []
    for value in (tasks, task):
        if value:
            if isinstance(value, str):
                selectors.extend(value.split(","))
            else:
                selectors.extend(value)

    if not selectors:
        return None

    requested_tasks = []
    for raw_selector in selectors:
        selector = str(raw_selector).strip()
        if not selector:
            continue

        if dataset == "longbench":
            group_name = LONGBENCH_TASK_GROUP_ALIASES.get(
                _normalize_task_selector(selector)
            )
            if group_name is not None:
                requested_tasks.extend(LONGBENCH_TASK_GROUPS[group_name])
                continue

        # An exact task name is still accepted for LongBench and other
        # datasets, preserving the previous --tasks behavior.
        requested_tasks.append(selector)

    # Deduplicate while preserving category/task ordering for stable filenames.
    return list(dict.fromkeys(requested_tasks)) or None


def _longbench_category_for_task(task_name: str):
    for category, category_tasks in LONGBENCH_TASK_GROUPS.items():
        if task_name in category_tasks:
            return category
    return None


def _print_longbench_category_summary(category: str, task_scores: dict):
    """Print task scores and the macro-average for one LongBench category."""
    category_label = LONGBENCH_TASK_GROUP_LABELS.get(category, category)
    print(f"\nLongBench category '{category_label}':", flush=True)
    for task_name, score in task_scores.items():
        if score is None:
            print(f"  {task_name}: N/A", flush=True)
        else:
            print(f"  {task_name}: {score:.2f}", flush=True)

    valid_scores = [score for score in task_scores.values() if score is not None]
    if valid_scores:
        category_average = sum(valid_scores) / len(valid_scores)
        print(f"  Category average: {category_average:.2f}", flush=True)
    else:
        print("  Category average: N/A", flush=True)



def evaluate(
    dataset: str = "ruler",
    # dataset: str = "longbench",
    data_dir: Optional[str] = "/datasets/SlowMov/ruler/4096/",
    model: str = "/models/Meta-Llama-3.1-8B-Instruct",
    device: Optional[str] = None,
    press_name: str = "efficient_ada_denfensive",
    compression_ratio: float = 0.75,
    snapkv_kernel_size: int = 7,
    fisher_window: int = 32,
    fisher_positions: int = 1,
    fisher_labels: int = 1,
    fisherlabel_samplemode: str = "multinomial",
    fisher_position_aggregation: str = "mean",
    score_mode: str = "separable",
    coupled_kernel_size: int = 1,
    coupled_pooling: str = "avg",
    first_stage_ratio: float = 0.5,
    fisher_seed: int = 42,
    attention_eps: float = 0.0,
    fraction: float = 0.2,
    tasks: Optional[str] = None,
    task: Optional[str] = None,
    max_new_tokens: Optional[int] = None,
    max_context_length: Optional[int] = None,
    compress_questions: bool = False,
):
    """
    Evaluate a model on a dataset using a press and save the results

    Parameters
    ----------
    dataset : str
        Dataset to evaluate
    data_dir : str, optional
        Subdirectory of the dataset to evaluate, by default None
    model : str, optional
        Model to use, by default "meta-llama/Meta-Llama-3.1-8B-Instruct"
    device : str, optional
        Model device, by default cuda:0 if available else cpu. For multi-GPU use "auto"
    press_name : str, optional
        Press to use (see PRESS_DICT), by default "expected_attention"
    compression_ratio : float, optional
        Compression ratio for the press, by default 0.1
    snapkv_kernel_size : int, optional
        Odd pooling kernel size used by SnapKV-based presses, by default 7
    fisher_window : int, optional
        Number of trailing differentiable tokens used by LogitKV, by default 32
    fisher_positions : int, optional
        Number of trailing logit positions used by LogitKV, by default 1
    fisher_labels : int, optional
        Number of independently sampled labels per LogitKV probe position, by default 1
    fisherlabel_samplemode : str, optional
        Fisher label selection mode: multinomial or top_fisherposition, by default multinomial
    fisher_position_aggregation : str, optional
        Aggregation across Fisher probe positions: mean or max, by default mean
    score_mode : str, optional
        LogitKV Stage-2 score: separable, coupled_diag, or coupled_full, by default separable
    coupled_kernel_size : int, optional
        Odd neighborhood-average kernel for coupled Q scores; 1 disables pooling, by default 1
    coupled_pooling : str, optional
        Coupled-Q neighborhood pooling: avg or max, by default avg
    first_stage_ratio : float, optional
        Fraction of the retained budget protected by attention-only Stage 1, by default 0.5
    fisher_seed : int, optional
        Sampling seed for LogitKV's Fisher probes, by default 42
    attention_eps : float, optional
        Stabilizer added to LogitKV's base attention score, by default 0
    max_new_tokens : int, optional
        Maximum number of new tokens to generate, by default use the default for the task (recommended)
    fraction : float, optional
        Fraction of the dataset to evaluate, by default 1.0
    tasks : str, optional
        Comma-separated exact task names or LongBench task groups to evaluate;
        None keeps every task, by default None. The supported LongBench groups
        are single_doc_qa, multidoc_qa, summarization, fewshot, synthetic, and
        code.
    task : str, optional
        Alias for tasks. It can be used to select one or more comma-separated
        LongBench groups, for example ``--task single_doc_qa,multidoc_qa``.
    max_context_length : int, optional
        Maximum number of tokens to use in the context. By default will use the maximum length supported by the model.
    compress_questions : bool, optional
        Whether to compress the questions as well, by default False
    """
    assert dataset in DATASET_DICT, f"No dataset found for {dataset}"
    assert dataset in SCORER_DICT, f"No scorer found for {dataset}"
    data_dir = str(data_dir) if data_dir else None
    requested_tasks = _resolve_requested_tasks(dataset, tasks=tasks, task=task)
    # Load press
    if press_name is not None:
        assert press_name in PRESS_DICT
        press = PRESS_DICT[press_name]
        assert snapkv_kernel_size > 0 and snapkv_kernel_size % 2 == 1, (
            "snapkv_kernel_size must be a positive odd integer"
        )
        snapkv_kernel_size_used = _configure_snapkv_kernel_size(press, snapkv_kernel_size)
        if isinstance(press, (DuoAttentionPress)):
            press.head_compression_ratio = compression_ratio
        else:
            press.compression_ratio = compression_ratio  # type:ignore[attr-definedif press is not None
        if isinstance(press, LogitKVPress):
            assert fisher_window > 0, "fisher_window must be positive"
            assert 0 < fisher_positions <= fisher_window, "fisher_positions must be in [1, fisher_window]"
            assert fisher_labels > 0, "fisher_labels must be positive"
            assert fisherlabel_samplemode in FISHER_LABEL_SAMPLE_MODES, (
                "fisherlabel_samplemode must be one of " + ", ".join(FISHER_LABEL_SAMPLE_MODES)
            )
            assert fisher_position_aggregation in ("mean", "max"), "invalid Fisher position aggregation"
            assert score_mode in ("separable", "coupled_diag", "coupled_full"), "invalid LogitKV score_mode"
            assert coupled_kernel_size > 0 and coupled_kernel_size % 2 == 1, (
                "coupled_kernel_size must be a positive odd integer"
            )
            assert coupled_pooling in ("avg", "max"), "invalid coupled pooling mode"
            assert 0 <= first_stage_ratio <= 1, "first_stage_ratio must be in [0, 1]"
            assert attention_eps >= 0, "attention_eps must be non-negative"
            assert score_mode == "separable" or attention_eps == 0, "coupled score modes require attention_eps=0"
            assert score_mode != "separable" or coupled_kernel_size == 1, (
                "coupled_kernel_size only applies to coupled score modes"
            )
            assert score_mode != "separable" or coupled_pooling == "avg", (
                "coupled_pooling only applies to coupled score modes"
            )
            press.fisher_window = fisher_window
            press.fisher_positions = fisher_positions
            press.fisher_labels = fisher_labels
            press.fisherlabel_samplemode = fisherlabel_samplemode
            press.fisher_position_aggregation = fisher_position_aggregation
            press.score_mode = score_mode
            press.coupled_kernel_size = coupled_kernel_size
            press.coupled_pooling = coupled_pooling
            press.first_stage_ratio = first_stage_ratio
            press.fisher_seed = fisher_seed
            press.attention_eps = attention_eps
    else:
        press = None
        snapkv_kernel_size_used = False

    if device is None:
        device = "cuda:7" if torch.cuda.is_available() else "cpu"

    # Keep result files separated by benchmark so LongBench and RULER runs do
    # not share one directory.
    save_dir = Path(__file__).parent / "results" / dataset
    save_dir.mkdir(parents=True, exist_ok=True)
    filename_parts = [
        dataset.replace("/", "--").split("--")[-1],
        model.replace("/", "--").split("--")[-1],
        press_name or "fullkv",
        f"cr{compression_ratio:g}",
    ]
    if snapkv_kernel_size_used:
        filename_parts.append(f"sk{snapkv_kernel_size}")
    if isinstance(press, LogitKVPress):
        filename_parts.extend(
            [
                f"fw{fisher_window}",
                f"positions{fisher_positions}",
                f"labels{fisher_labels}",
                f"mode{score_mode}",
            ]
        )
        if fisher_position_aggregation != "mean":
            filename_parts.append(f"pagg{fisher_position_aggregation}")
        if score_mode != "separable":
            filename_parts.append(f"ck{coupled_kernel_size}")
            if coupled_pooling != "avg":
                filename_parts.append(f"cp{coupled_pooling}")
        filename_parts.append(f"sr{first_stage_ratio:g}")
        filename_parts.extend([f"fs{fisher_seed}", f"ae{attention_eps:g}"])
        fisherlabel_sample_tag = {
            "multinomial": "flsmmulti",
            "top_fisherposition": "flsmtop",
        }[fisherlabel_samplemode]
        filename_parts.append(fisherlabel_sample_tag)
    filename_parts.append(f"frac{fraction:.2f}")
    if requested_tasks:
        filename_parts.append(f"tasks{'+'.join(requested_tasks)}")
    if max_context_length is not None:
        filename_parts.append(f"max_context{max_context_length}")
    if compress_questions:
        filename_parts.append("compressed_questions")
    save_filename = save_dir / ("__".join(filename_parts) + ".csv")
    print("try save to:", save_filename)
    if save_filename.exists():
        logger.warning(f"Results already exist at {save_filename}")
        print("Results already exist at", save_filename)
        exit()

    # Load dataframe
    try:
        print("Loading from disk, data_dir:", data_dir)
        df = load_from_disk(data_dir).to_pandas()
    except Exception as e:
        print(f"Failed to load from disk: {e}")
        exit()

    if fraction < 1.0:
        # Stratified sampling by task category
        sampled_dfs = []
        for task_name, task_df in df.groupby("task"):
            sampled_task_df = task_df.sample(frac=fraction, random_state=42)
            sampled_dfs.append(sampled_task_df)
        df = pd.concat(sampled_dfs)

    if requested_tasks:
        available_tasks = set(df["task"].unique())
        missing_tasks = sorted(set(requested_tasks) - available_tasks)
        assert not missing_tasks, f"Requested tasks are unavailable: {missing_tasks}"
        df = df[df["task"].isin(requested_tasks)].copy()

    if compress_questions:
        df["context"] = df["context"] + df["question"]
        df["question"] = "\n"

    # Initialize pipeline with the correct attention implementation
    model_kwargs = {}
    if isinstance(press, ObservedAttentionPress):
        model_kwargs = {"attn_implementation": "eager"}
    # Support AdaKV
    elif isinstance(press, EfficientAdaScorerPress):
        replace_var_flash_attn(model_name=model)
    elif isinstance(press, EfficientAdaGlobalScorerPress):
        replace_var_flash_attn(model_name=model)
    else:
        try:
            import flash_attn  # noqa: F401

            model_kwargs = {"attn_implementation": "flash_attention_2"}
        except ImportError:
            pass

    model_kwargs["torch_dtype"] = "auto"
    if device == "auto":
        pipe = pipeline("kv-press-text-generation", model=model, device_map="auto", model_kwargs=model_kwargs)
    else:
        pipe = pipeline("kv-press-text-generation", model=model, device=device, model_kwargs=model_kwargs)

    print("model dtype: ", pipe.model.dtype, flush=True)
    # Run pipeline on each context
    df["predicted_answer"] = None
    assert all(df.groupby("context")["answer_prefix"].nunique() == 1)

    if dataset == "longbench": 
        # evalutated_tasks = ["qasper"]
        # evalutated_tasks = ["hotpotqa"]
        evalutated_tasks = None # Test all
    elif dataset == "ruler":
        # evalutated_tasks = ["niah_multivalue"]
        evalutated_tasks = None # Test all
    else:
        evalutated_tasks = None

    scorer = SCORER_DICT[dataset]
    task_groups = df.groupby("task", sort=False)
    category_task_order = {}
    category_task_scores = {}
    printed_categories = set()
    if dataset == "longbench":
        for task_name in task_groups.groups:
            category = _longbench_category_for_task(task_name)
            if category is not None:
                category_task_order.setdefault(category, []).append(task_name)
    total_context_groups = df.groupby(["task", "context"], sort=False).ngroups
    # Count failed context groups across the entire evaluation run. This must
    # not be reset inside the per-context loop.
    Failure_count = 0
    with tqdm(total=total_context_groups) as progress:
        for task_name, task_df in task_groups:
            # skip specific tasks, which are not in the task_names
            if evalutated_tasks is not None and task_name not in evalutated_tasks:
                progress.update(task_df["context"].nunique())
                continue

            for context, df_ in task_df.groupby("context", sort=False):
                chat_template_bak = pipe.tokenizer.chat_template
                bos_bak = pipe.tokenizer.bos_token
                gen_config_eos_id_bak = pipe.model.generation_config.eos_token_id

                if task_name in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]:
                    pipe.tokenizer.chat_template = None
                    pipe.tokenizer.bos_token = ""
                    if task_name in ["samsum"]:
                        pipe.model.generation_config.eos_token_id = [
                            pipe.tokenizer.eos_token_id,
                            pipe.tokenizer.encode("\n", add_special_tokens=False)[-1],
                        ]

                questions = df_["question"].to_list()
                max_new_tokens_ = max_new_tokens if max_new_tokens is not None else df_["max_new_tokens"].iloc[0]
                answer_prefix = df_["answer_prefix"].iloc[0]
                try:
                    output = pipe(
                        context,
                        questions=questions,
                        answer_prefix=answer_prefix,
                        press=press,
                        max_new_tokens=max_new_tokens_,
                        max_context_length=max_context_length,
                    )
                except Exception as e:
                    print("An error occurred:", e)
                    output = {"answers": "Failure:" + str(e)}
                    Failure_count += 1

                df.loc[df_.index, "predicted_answer"] = output["answers"]
                if press:
                    df.loc[df_.index, "compression_ratio"] = press.compression_ratio  # type:ignore[attr-defined]
                else:
                    df.loc[df_.index, "compression_ratio"] = 0  # type:ignore[attr-defined]

                # restore chat template
                pipe.tokenizer.chat_template = chat_template_bak
                pipe.tokenizer.bos_token = bos_bak
                pipe.model.generation_config.eos_token_id = gen_config_eos_id_bak
                progress.update(1)

            # Score the complete task after all of its contexts have finished.
            completed_task_df = df.loc[task_df.index].copy()
            task_score = None
            try:
                task_metrics = scorer(completed_task_df)
                print(f"\nTask '{task_name}' metrics: {task_metrics}", flush=True)
                if isinstance(task_metrics, dict):
                    task_score = task_metrics.get(task_name)
            except Exception as e:
                print(f"\nFailed to calculate metrics for task '{task_name}': {e}", flush=True)

            if dataset == "longbench":
                category = _longbench_category_for_task(task_name)
                if category is not None and category not in printed_categories:
                    category_task_scores.setdefault(category, {})[task_name] = task_score
                    selected_category_tasks = category_task_order[category]
                    if all(
                        selected_task in category_task_scores[category]
                        for selected_task in selected_category_tasks
                    ):
                        _print_longbench_category_summary(
                            category, category_task_scores[category]
                        )
                        printed_categories.add(category)

    # Save answers
    df[["predicted_answer", "compression_ratio"]].to_csv(str(save_filename), index=False)

    print("Saving DataFrame to", save_filename)

    df.to_csv(str(save_filename).replace(".csv", "_df.csv"), index=False)
    # Calculate metrics
    metrics = scorer(df)
    with open(str(save_filename).replace(".csv", ".json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"Average compression ratio: {df['compression_ratio'].mean():.2f}")
    print(f"Failure count: {Failure_count}")
    print(metrics)


if __name__ == "__main__":
    Fire(evaluate)
