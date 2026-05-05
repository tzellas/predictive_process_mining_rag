import re
import pandas as pd
from pathlib import Path
from llm_api import api_call, stop_ollama_model
from retrieval import Retrieval


def normalize_label(label: str | None) -> str | None:
    if label is None:
        return None
    return label.replace("\\ ", " ").strip().strip("\"'").casefold()


def extract_prediction(output_text: str) -> str | None:
    match = re.search(r"<answer>(.*?)</answer>", output_text, re.DOTALL)
    if match:
        prediction = normalize_label(match.group(1))
        if prediction in {None, "", "none", "null", "n/a", "na"}:
            return None
        return prediction
    return None


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _per_label_counts(y_true: list[str], y_pred: list[str | None], label: str) -> tuple[int, int, int]:
    tp = fp = fn = 0
    for gold, pred in zip(y_true, y_pred):
        if pred == label and gold == label:
            tp += 1
        elif pred == label and gold != label:
            fp += 1
        elif pred != label and gold == label:
            fn += 1
    return tp, fp, fn


def compute_classification_metrics(y_true: list[str], y_pred: list[str | None]) -> dict:
    labels = sorted(set(y_true) | {pred for pred in y_pred if pred is not None})
    total = len(y_true)
    correct = sum(1 for gold, pred in zip(y_true, y_pred) if gold == pred)

    macro_precision = 0.0
    macro_recall = 0.0
    macro_f1 = 0.0

    for label in labels:
        tp, fp, fn = _per_label_counts(y_true, y_pred, label)

        precision = _safe_divide(tp, tp + fp)
        recall = _safe_divide(tp, tp + fn)
        f1 = _safe_divide(2 * precision * recall, precision + recall)

        macro_precision += precision
        macro_recall += recall
        macro_f1 += f1

    label_count = len(labels) if labels else 1

    return {
        "total_samples": total,
        "correct_predictions": correct,
        "accuracy": _safe_divide(correct, total),
        "precision_macro": macro_precision / label_count,
        "recall_macro": macro_recall / label_count,
        "f1_macro": macro_f1 / label_count,
    }


def prefix_length(prefix: str) -> int:
    prefix_without_values = prefix.split(" - Values: ", 1)[0]
    activities = [activity.strip() for activity in prefix_without_values.split(",") if activity.strip()]
    return len(activities)


def _bucket_label(length: int, boundaries: tuple[int, ...]) -> str:
    lower_bound = 1
    for boundary in boundaries:
        if length <= boundary:
            return f"{lower_bound}-{boundary}"
        lower_bound = boundary + 1
    return f"{lower_bound}+"


def compute_earlyness_metrics(
    rows: list[dict],
    boundaries: tuple[int, ...] = (5, 10, 20, 30),
) -> dict:
    bucket_rows: dict[str, list[dict]] = {}

    for row in rows:
        bucket = _bucket_label(row["prefix_length"], boundaries)
        bucket_rows.setdefault(bucket, []).append(row)

    earlyness = {
        "boundaries": list(boundaries),
        "buckets": {},
    }

    total_rows = len(rows)
    for bucket, bucket_items in bucket_rows.items():
        y_true = [item["gold_label"] for item in bucket_items]
        y_pred = [item["prediction"] for item in bucket_items]
        bucket_metrics = compute_classification_metrics(y_true, y_pred)
        earlyness["buckets"][bucket] = {
            **bucket_metrics,
            "share_of_total": _safe_divide(len(bucket_items), total_rows),
        }

    return earlyness


def basic_metrics(
    retrieval: Retrieval,
    top_k: int = 20,
    max_rows: int | None = None,
    use_process_guidance: bool = True,
    llm_model: str | None = None,
    prompt_variant: str | None = None,
    keep_individual_predictions: bool = False,
    earlyness_boundaries: tuple[int, ...] = (5, 10, 20, 30),
) -> dict:
    test_df = pd.read_csv(retrieval.test_set)
    if max_rows is not None:
        # Keep sampling deterministic but continue until we collect max_rows valid predictions.
        # This skips invalid/None model outputs instead of counting them in the requested budget.
        test_df = test_df.sample(frac=1, random_state=7).copy()
        target_valid_predictions = min(max_rows, len(test_df))
    else:
        target_valid_predictions = len(test_df)

    y_true: list[str] = []
    y_pred: list[str | None] = []
    individual_predictions: list[dict] = []
    skipped_invalid_predictions = 0

    for index, row in enumerate(test_df.itertuples(index=False), start=1):
        prefix = row.prefix
        gold_label = normalize_label(row.prediction)

        context, _ = retrieval.retrieve_similar_prefixes(prefix, top_k)
        dataset_name = retrieval.variant_name
        output_text = api_call(
            context,
            prefix,
            dataset_name=dataset_name,
            use_process_guidance=use_process_guidance,
            llm_model=llm_model,
            prompt_variant=prompt_variant,
        )
        predicted_label = extract_prediction(output_text)

        if predicted_label is not None:
            y_true.append(gold_label)
            y_pred.append(predicted_label)
            individual_predictions.append(
                {
                    "row_index": index,
                    "prefix_length": prefix_length(prefix),
                    "prefix": prefix,
                    "prediction": predicted_label,
                    "gold_label": gold_label,
                    "is_correct": predicted_label == gold_label,
                    "raw_output": output_text,
                }
            )
        else:
            skipped_invalid_predictions += 1

        print(
            f"[{index}/{len(test_df)}] "
            f"predicted={predicted_label} | gold={gold_label}"
        )

        if len(y_true) >= target_valid_predictions:
            break

    metrics = compute_classification_metrics(y_true, y_pred)
    metrics["dataset"] = retrieval.variant_name
    metrics["model_id"] = retrieval.model_id
    metrics["llm"] = llm_model
    metrics["prompt_variant"] = prompt_variant
    metrics["top_k"] = top_k
    metrics["reranker_model"] = retrieval.reranker_model_id
    metrics["rerank_pool_k"] = retrieval.rerank_pool_k
    metrics["skipped_invalid_predictions"] = skipped_invalid_predictions
    metrics["earlyness"] = compute_earlyness_metrics(
        rows=individual_predictions,
        boundaries=earlyness_boundaries,
    )
    if keep_individual_predictions:
        metrics["individual_predictions"] = individual_predictions
    return metrics


def mean_metrics(dataset_reports: list[dict]) -> dict:
    if not dataset_reports:
        raise ValueError("dataset_reports cannot be empty")

    metric_keys = [
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
    ]

    mean_report = {
        "dataset_count": len(dataset_reports),
        "datasets": [report["dataset"] for report in dataset_reports],
    }

    for key in metric_keys:
        mean_report[f"mean_{key}"] = sum(report[key] for report in dataset_reports) / len(dataset_reports)

    mean_report["total_samples"] = sum(report["total_samples"] for report in dataset_reports)
    mean_report["total_correct_predictions"] = sum(report["correct_predictions"] for report in dataset_reports)
    mean_report["micro_accuracy_over_all_samples"] = _safe_divide(
        mean_report["total_correct_predictions"],
        mean_report["total_samples"],
    )

    return mean_report


def evaluate_multiple_datasets(
    retrieval_setups: list[Retrieval],
    top_k: int = 20,
    max_rows: int | None = None,
    use_process_guidance: bool = True,
    llm_model: str | None = None,
    prompt_variant: str | None = None,
    keep_individual_predictions: bool = False,
    earlyness_boundaries: tuple[int, ...] = (5, 10, 20, 30),
) -> dict:
    dataset_reports: list[dict] = []
    total_setups = len(retrieval_setups)

    for index, retrieval in enumerate(retrieval_setups):
        report = basic_metrics(
            retrieval=retrieval,
            top_k=top_k,
            max_rows=max_rows,
            use_process_guidance=use_process_guidance,
            llm_model=llm_model,
            prompt_variant=prompt_variant,
            keep_individual_predictions=keep_individual_predictions,
            earlyness_boundaries=earlyness_boundaries,
        )
        dataset_reports.append(report)

        if index < total_setups - 1 and llm_model:
            stop_ollama_model(llm_model)

    return {
        "dataset_reports": dataset_reports,
        "mean_report": mean_metrics(dataset_reports),
    }
