import json
from os import name
import os.path
from pathlib import Path
import pandas as pd


LONG_BENCH_CATEGORIES = {
    "SingleDoc QA": ["narrativeqa", "qasper", "multifieldqa_en"],
    "multidoc QA": ["hotpotqa", "2wikimqa", "musique"],
    "summarization": ["gov_report", "qmsum", "multi_news"],
    "fewshot": ["trec", "triviaqa", "samsum"],
    "synthetic": ["passage_count", "passage_retrieval_en"],
    "code": ["lcc", "repobench-p"],
}

AGGREGATE_KEYS = {"average_score", "macro_average"}


def calculate_category_averages(score_dict, categories):
    """Calculate one macro-average per category from task-level scores."""
    category_scores = {}
    for category, tasks in categories.items():
        scores = [score_dict[task] for task in tasks if task in score_dict]
        category_scores[category] = sum(scores) / len(scores) if scores else None
    return category_scores


# Function to calculate the average score from a JSON file
def calculate_average_score(file_path):
    try:
        # Read the JSON file
        with open(file_path, 'r') as file:
            data = json.load(file)

        # Initialize variables for total score and count
        total_score = 0
        count = 0
        score_dict = {}
        # Iterate through the JSON data to sum scores
        for key, value in data.items():
            # Ignore aggregate values that may have been added to a result JSON
            # by evaluation/results/add_macro_average.py.
            if key in AGGREGATE_KEYS:
                continue
            # Handle both direct float values and dict with "string_match" key
            if isinstance(value, dict):
                score = value.get("string_match", 0)
            elif isinstance(value, (int, float)):
                score = value
            else:
                score = 0
            score_dict[key] = score
            total_score += score
            count += 1

        # Calculate the average score
        average_score = total_score / count if count > 0 else 0
        score_dict['average_score'] = average_score
        # Print the results
        print(f"JSON Name: {file_path}")
        print(f"Average Score: {average_score:.2f}")
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except json.JSONDecodeError:
        print("Invalid JSON file.")
    return file_path, score_dict


def process_dataset(json_files, dataset_name):
    """Process a specific dataset and return the DataFrame"""
    print(f"\n{'='*60}")
    print(f"Processing {dataset_name.upper()} dataset")
    print(f"{'='*60}")
    
    pd_list = []
    all_keys = set()
    file_score_dicts = []
    
    categories = LONG_BENCH_CATEGORIES if dataset_name.lower() == "longbench" else {}

    # First pass: collect all keys and scores
    for file in json_files:
        file_name, score_dict = calculate_average_score(file)
        score_dict.update(calculate_category_averages(score_dict, categories))
        file_score_dicts.append((Path(file).name, score_dict))
        all_keys.update(score_dict.keys())
    
    if not all_keys:
        print(f"No data found for {dataset_name}")
        return None
    
    # Keep task columns first, category averages next, and the overall average
    # as the final column.
    category_keys = [category for category in categories if category in all_keys]
    task_keys = sorted(
        key
        for key in all_keys
        if key not in AGGREGATE_KEYS and key not in category_keys
    )
    sorted_keys = task_keys + category_keys
    if "average_score" in all_keys:
        sorted_keys.append("average_score")
    
    # Build data rows
    for file_name, score_dict in file_score_dicts:
        score_list = [score_dict.get(key, None) for key in sorted_keys]
        score_list.insert(0, file_name)
        pd_list.append(score_list)
    
    columns = ['File Name'] + sorted_keys
    df = pd.DataFrame(pd_list, columns=columns)
    return df


if __name__ == '__main__':
    results_root = Path(__file__).resolve().parent
    root_json_files = sorted(results_root.glob('*.json'))

    print('Result directories:')
    print(f"  LongBench: {results_root / 'longbench'}")
    print(f"  RULER: {results_root / 'ruler'}")

    # Include legacy JSON files that were written directly to results/ so the
    # statistics script remains useful after the result-directory migration.
    dataset_files = {}
    for dataset_name in ('longbench', 'ruler'):
        nested_files = sorted((results_root / dataset_name).glob('*.json'))
        legacy_files = [
            path for path in root_json_files
            if dataset_name in path.name.lower()
        ]
        dataset_files[dataset_name] = nested_files + legacy_files

    other_files = [
        path for path in root_json_files
        if not any(dataset_name in path.name.lower() for dataset_name in ('longbench', 'ruler'))
    ]

    print(
        f"\nFound {len(dataset_files['longbench'])} LongBench files, "
        f"{len(dataset_files['ruler'])} RULER files, "
        f"{len(other_files)} other files"
    )

    for dataset_name in ('longbench', 'ruler'):
        json_files = dataset_files[dataset_name]
        if not json_files:
            print(f"\nNo {dataset_name.upper()} JSON files found")
            continue

        dataframe = process_dataset(json_files, dataset_name)
        if dataframe is not None:
            output_dir = results_root / dataset_name
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_dir / f'{dataset_name}_scores.xlsx'
            dataframe.to_excel(output_file, index=False)
            print(f"\n{dataset_name.upper()} results saved to '{output_file}'")

    # Process legacy result files from other datasets, if any.
    if other_files:
        df_other = process_dataset(other_files, 'other')
        if df_other is not None:
            output_file = results_root / 'other_scores.xlsx'
            df_other.to_excel(output_file, index=False)
            print(f"\nOther results saved to '{output_file}'")
