import csv
import json
import time
from datetime import datetime, UTC
from pathlib import Path

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
