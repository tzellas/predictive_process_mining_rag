import csv
import json
import ast
import re
import time
from datetime import datetime, UTC
from pathlib import Path


ATTRIBUTE_LABEL_MAP = {
    "titi": "time:timestamp",
    "orre": "org:resource",
    "orro": "org:role",
    "litr": "lifecycle:transition",
    "id": "event identifier",
    "Us": "user",
    "Cunewo(E": "net value",
}

def same_lines_csvs(path1, path2):
    with open(path1, newline='', encoding="utf-8") as f1, \
        open(path2, newline='', encoding="utf-8") as f2:
        rows1 = list(csv.reader(f1))
        rows2 = list(csv.reader(f2))
        same_first_column = sorted(row[0] for row in rows1) == sorted(row[0] for row in rows2)
        same_rows = sorted(rows1) == sorted(rows2)
        print(f"first column {same_first_column}, all csv {same_rows}")
        return same_first_column, same_rows


def count_csv_rows(path: str | Path) -> int:
    with open(path, newline="", encoding="utf-8") as f:
        return sum(1 for _ in csv.DictReader(f))


def format_duration(seconds: float) -> str:
    total_seconds = int(round(seconds))
    minutes, secs = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    return f"{minutes}m {secs}s"


def bytes_to_mb(num_bytes: int) -> float:
    return round(num_bytes / (1024 * 1024), 3)


def build_attribute_labels_json(dataset_dir: str | Path) -> Path:
    dataset_dir = Path(dataset_dir)
    labels_path = dataset_dir / "attribute_labels.json"
    keys: set[str] = set()

    for csv_name in ("retrieval.csv", "test.csv"):
        csv_path = dataset_dir / csv_name
        if not csv_path.exists():
            continue

        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                prefix = row.get("prefix", "")
                if " - Values: " not in prefix:
                    continue

                values_text = prefix.rsplit(" - Values: ", 1)[1]
                try:
                    values = ast.literal_eval(values_text)
                except (ValueError, SyntaxError):
                    continue

                if isinstance(values, dict):
                    keys.update(str(key) for key in values.keys())

    labels = {
        key: ATTRIBUTE_LABEL_MAP.get(key, key)
        for key in sorted(keys)
    }

    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2, ensure_ascii=False)

    return labels_path


def append_report_history(
    report: dict,
    variant_key: str,
    variant_config: dict,
    report_path: str | Path,
) -> Path:
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    if report_path.exists():
        with open(report_path, encoding="utf-8") as f:
            history = json.load(f)
    else:
        history = {}

    if variant_key not in history:
        history[variant_key] = {
            "config": variant_config,
            "runs": [],
        }

    history[variant_key]["config"] = variant_config
    history[variant_key]["runs"] = [report]

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    return report_path


def dump_evaluation_results(
    runs: list[dict],
    results_path: str | Path = "../data/evaluation_results/evaluation_runs.json",
) -> Path:
    results_path = Path(results_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)

    if results_path.exists():
        with open(results_path, "r", encoding="utf-8") as f:
            saved_runs = json.load(f)
    else:
        saved_runs = {}

    if isinstance(saved_runs, list):
        saved_runs = _migrate_legacy_evaluation_runs(saved_runs)

    for run in runs:
        timestamp = run.get("timestamp")
        run_label = run.get("run_label")
        top_k = run.get("top_k")
        max_rows = run.get("max_rows")
        use_process_guidance = run.get("use_process_guidance")
        reranker_model = run.get("reranker_model")
        rerank_pool_k = run.get("rerank_pool_k")

        for report in run.get("reports", []):
            config = _build_evaluation_config(
                dataset=report.get("dataset"),
                model_id=report.get("model_id"),
                llm=report.get("llm"),
                prompt_variant=report.get("prompt_variant"),
                top_k=top_k if top_k is not None else report.get("top_k"),
                max_rows=max_rows,
                use_process_guidance=use_process_guidance,
                reranker_model=reranker_model if reranker_model is not None else report.get("reranker_model"),
                rerank_pool_k=rerank_pool_k if rerank_pool_k is not None else report.get("rerank_pool_k"),
            )
            config_key = _evaluation_config_key(config)

            if config_key not in saved_runs:
                saved_runs[config_key] = {
                    "config": config,
                    "runs": [],
                }

            run_entry = {
                "timestamp": timestamp,
                "run_label": run_label,
                "metrics": report,
            }
            saved_runs[config_key]["runs"].append(run_entry)

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(saved_runs, f, indent=2, ensure_ascii=False)

    return results_path


def _parse_variant_config(dataset: str | None) -> dict[str, int | None]:
    if not dataset:
        return {"base": None, "gap": None, "m": None}

    match = re.match(r"^(?:last_(\d+)_)?b(\d+)_g(\d+)_(.+)$", dataset)
    if not match:
        return {"base": None, "gap": None, "m": None}

    m_raw, base_raw, gap_raw, _ = match.groups()
    return {
        "base": int(base_raw),
        "gap": int(gap_raw),
        "m": int(m_raw) if m_raw is not None else 1,
    }


def _build_evaluation_config(
    dataset: str | None,
    model_id: str | None,
    llm: str | None,
    prompt_variant: str | None,
    top_k: int | None,
    max_rows: int | None,
    use_process_guidance: bool | None,
    reranker_model: str | None = None,
    rerank_pool_k: int | None = None,
) -> dict:
    variant = _parse_variant_config(dataset)
    split_type = "row_level" if dataset and dataset.endswith("_row") else "trace_level"
    return {
        "dataset": dataset,
        "base": variant["base"],
        "gap": variant["gap"],
        "m": variant["m"],
        "top_k": top_k,
        "max_rows": max_rows,
        "model_id": model_id,
        "llm": llm,
        "prompt_variant": prompt_variant,
        "use_process_guidance": use_process_guidance,
        "reranker_model": reranker_model,
        "rerank_pool_k": rerank_pool_k,
        "split_type": split_type,
    }


def _evaluation_config_key(config: dict) -> str:
    return (
        f"dataset={config.get('dataset')}|"
        f"base={config.get('base')}|"
        f"gap={config.get('gap')}|"
        f"m={config.get('m')}|"
        f"top_k={config.get('top_k')}|"
        f"max_rows={config.get('max_rows')}|"
        f"model_id={config.get('model_id')}|"
        f"llm={config.get('llm')}|"
        f"prompt_variant={config.get('prompt_variant')}|"
        f"use_process_guidance={config.get('use_process_guidance')}|"
        f"reranker_model={config.get('reranker_model')}|"
        f"rerank_pool_k={config.get('rerank_pool_k')}"
    )


def _migrate_legacy_evaluation_runs(legacy_runs: list[dict]) -> dict:
    migrated: dict[str, dict] = {}

    for run in legacy_runs:
        timestamp = run.get("timestamp")
        run_label = run.get("run_label")
        top_k = run.get("top_k")
        max_rows = run.get("max_rows")
        use_process_guidance = run.get("use_process_guidance")
        reranker_model = run.get("reranker_model")
        rerank_pool_k = run.get("rerank_pool_k")

        for report in run.get("reports", []):
            config = _build_evaluation_config(
                dataset=report.get("dataset"),
                model_id=report.get("model_id"),
                llm=report.get("llm"),
                prompt_variant=report.get("prompt_variant"),
                top_k=top_k if top_k is not None else report.get("top_k"),
                max_rows=max_rows,
                use_process_guidance=use_process_guidance,
                reranker_model=reranker_model if reranker_model is not None else report.get("reranker_model"),
                rerank_pool_k=rerank_pool_k if rerank_pool_k is not None else report.get("rerank_pool_k"),
            )
            config_key = _evaluation_config_key(config)

            if config_key not in migrated:
                migrated[config_key] = {
                    "config": config,
                    "runs": [],
                }

            migrated[config_key]["runs"].append(
                {
                    "timestamp": timestamp,
                    "run_label": run_label,
                    "metrics": report,
                }
            )

    return migrated


def execution_report(
    datasets: list[str | Path],
    run_fn,
    base: int,
    gap: int,
    m: int,
    report_path: str | Path | None = None,
) -> dict:
    variant_key = f"last_{m}_b{base}_g{gap}" if m > 1 else f"b{base}_g{gap}"
    variant_config = {"base": base, "gap": gap, "m": m}
    total_wall_start = time.perf_counter()
    total_cpu_start = time.process_time()
    report = {
        "dataset_count": len(datasets),
        "datasets": {},
    }

    for dataset in datasets:
        dataset_path = Path(dataset)
        dataset_name = dataset_path.stem
        wall_start = time.perf_counter()
        cpu_start = time.process_time()

        try:
            output_path = Path(run_fn(dataset))
            error = None
        except Exception as exc:
            output_path = None
            error = str(exc)

        wall_elapsed = time.perf_counter() - wall_start
        cpu_elapsed = time.process_time() - cpu_start
        dataset_report = {
            "input_path": str(dataset_path),
            "elapsed": format_duration(wall_elapsed),
            "cpu_time": format_duration(cpu_elapsed),
        }

        if output_path is not None:
            if output_path.exists():
                dataset_report["output_size_mb"] = bytes_to_mb(output_path.stat().st_size)
                dataset_report["output_rows"] = count_csv_rows(output_path)

        if error is not None:
            dataset_report["error"] = error

        report["datasets"][dataset_name] = dataset_report

    report["total_elapsed"] = format_duration(time.perf_counter() - total_wall_start)
    report["total_cpu_time"] = format_duration(time.process_time() - total_cpu_start)

    if report_path is not None:
        saved_path = append_report_history(
            report=report,
            variant_key=variant_key,
            variant_config=variant_config,
            report_path=report_path,
        )
        report["saved_report_path"] = str(saved_path)

    print(json.dumps(report, indent=2))
    return report
