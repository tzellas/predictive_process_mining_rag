import argparse
from datetime import datetime
from pathlib import Path

from evaluation import evaluate_multiple_datasets
from event_log_preprocessing import process_log
from process_discovery import discover_log_for_variant
from retrieval import Retrieval
from utils import build_attribute_labels_json, dump_evaluation_results


REPO_ROOT = Path(__file__).resolve().parent.parent


def _resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def _resolve_dataset_dir(dataset_text: str) -> Path:
    path = Path(dataset_text)
    if path.is_absolute():
        return path

    repo_candidate = (REPO_ROOT / path).resolve()
    if repo_candidate.exists():
        return repo_candidate

    data_candidate = (REPO_ROOT / "data" / dataset_text).resolve()
    return data_candidate


def _print_reports(run_label: str, reports: list[dict]) -> None:
    print(run_label)
    for report in reports:
        print(report["model_id"])
        print(report["llm"])
        print(report["prompt_variant"])
        print(report["dataset"])
        print(report["accuracy"])
        print(report["precision_macro"])
        print(report["recall_macro"])
        print(report["f1_macro"])
        if "earlyness" in report:
            print(report["earlyness"])
        print()


def preprocess_command(args: argparse.Namespace) -> None:
    embedding_models = args.embedding_models or []

    for dataset_arg in args.datasets:
        dataset_xes_path = _resolve_path(dataset_arg)
        print(
            f"Processing {dataset_xes_path.name} "
            f"with b={args.base}, g={args.gap}, m={args.m}, split={args.split_mode}"
        )
        retrieval_csv_path, test_csv_path, train_xes_path, _ = process_log(
            dataset_xes=dataset_xes_path,
            shard_output_dir=_resolve_path(args.shard_output_dir) if args.shard_output_dir else None,
            base=args.base,
            gap=args.gap,
            trace_identifier=args.trace_identifier,
            test_set_proportion=args.test_set_proportion,
            m=args.m,
            split_mode=args.split_mode,
        )

        discovered_process_path = None
        if args.discovery and train_xes_path is not None:
            discovered_process_path = discover_log_for_variant(
                train_xes_path=train_xes_path,
                original_xes_path=dataset_xes_path,
                base=args.base,
                gap=args.gap,
                m=args.m,
                split_mode=args.split_mode,
            )
        elif args.discovery and args.split_mode == "row":
            print("Skipping discovery for row split mode because there is no train-only XES split.")

        labels_path = None
        if args.attribute_labels:
            labels_path = build_attribute_labels_json(retrieval_csv_path.parent)

        if args.store_embeddings:
            for model_id in embedding_models:
                print(f"Building collection for {retrieval_csv_path.parent.name} with {model_id}")
                retrieval = Retrieval(
                    dataset=retrieval_csv_path,
                    test_set=test_csv_path,
                    model_id=model_id,
                )
                retrieval.store_embeddings(batch_size=args.batch_size)

        print(f"retrieval.csv: {retrieval_csv_path}")
        print(f"test.csv: {test_csv_path}")
        if discovered_process_path is not None:
            print(f"discovered_process.txt: {discovered_process_path}")
        if labels_path is not None:
            print(f"attribute_labels.json: {labels_path}")
        print()


def eval_command(args: argparse.Namespace) -> None:
    dataset_dirs = [_resolve_dataset_dir(dataset_arg) for dataset_arg in args.datasets]
    if args.reranker_model and args.rerank_pool_k is None:
        raise ValueError("--rerank-pool-k is required when --reranker-model is provided.")
    if args.rerank_pool_k is not None and not args.reranker_model:
        raise ValueError("--reranker-model is required when --rerank-pool-k is provided.")
    if args.rerank_pool_k is not None and args.rerank_pool_k < args.top_k:
        raise ValueError("--rerank-pool-k must be >= --top-k.")

    retrieval_setups: list[Retrieval] = []
    for dataset_dir in dataset_dirs:
        retrieval_csv_path = dataset_dir / "retrieval.csv"
        test_csv_path = dataset_dir / "test.csv"
        for model_id in args.embedding_models:
            retrieval_setups.append(
                Retrieval(
                    dataset=retrieval_csv_path,
                    test_set=test_csv_path,
                    model_id=model_id,
                    reranker_model_id=args.reranker_model,
                    rerank_pool_k=args.rerank_pool_k,
                )
            )

    evaluation_report = evaluate_multiple_datasets(
        retrieval_setups,
        top_k=args.top_k,
        max_rows=args.max_rows,
        use_process_guidance=args.use_process_guidance,
        llm_model=args.llm_model,
        prompt_variant=args.prompt_variant,
        keep_individual_predictions=args.keep_individual_predictions,
        earlyness_boundaries=tuple(args.earlyness_buckets),
    )

    run_label = args.run_label
    _print_reports(run_label, evaluation_report["dataset_reports"])

    evaluation_runs = [
        {
            "timestamp": datetime.now().isoformat(),
            "run_label": run_label,
            "datasets": [report["dataset"] for report in evaluation_report["dataset_reports"]],
            "top_k": args.top_k,
            "max_rows": args.max_rows,
            "use_process_guidance": args.use_process_guidance,
            "llm": args.llm_model,
            "prompt_variant": args.prompt_variant,
            "reranker_model": args.reranker_model,
            "rerank_pool_k": args.rerank_pool_k,
            "keep_individual_predictions": args.keep_individual_predictions,
            "earlyness_buckets": args.earlyness_buckets,
            "reports": evaluation_report["dataset_reports"],
        }
    ]

    results_path = dump_evaluation_results(evaluation_runs, results_path=args.results_path)
    print(f"Saved evaluation results to {results_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CLI for preprocessing and evaluation workflows.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    preprocess_parser = subparsers.add_parser("preprocess", help="Preprocess XES logs and optionally build collections.")
    preprocess_parser.add_argument("--datasets", nargs="+", required=True, help="XES paths to preprocess.")
    preprocess_parser.add_argument("--base", type=int, default=1)
    preprocess_parser.add_argument("--gap", type=int, default=3)
    preprocess_parser.add_argument("--m", type=int, default=1)
    preprocess_parser.add_argument("--trace-identifier", default="case:concept:name")
    preprocess_parser.add_argument("--test-set-proportion", type=float, default=0.3)
    preprocess_parser.add_argument("--split-mode", choices=["trace", "row"], default="trace")
    preprocess_parser.add_argument("--shard-output-dir")
    preprocess_parser.add_argument("--no-discovery", dest="discovery", action="store_false")
    preprocess_parser.add_argument("--no-attribute-labels", dest="attribute_labels", action="store_false")
    preprocess_parser.add_argument("--store-embeddings", action="store_true")
    preprocess_parser.add_argument("--embedding-models", nargs="+", choices=["bge", "minilm"], default=["bge", "minilm"])
    preprocess_parser.add_argument("--batch-size", type=int, default=32)
    preprocess_parser.set_defaults(discovery=True, attribute_labels=True, func=preprocess_command)

    eval_parser = subparsers.add_parser("eval", help="Run evaluation on existing dataset folders.")
    eval_parser.add_argument("--datasets", nargs="+", required=True, help="Dataset directories or dataset names under data/.")
    eval_parser.add_argument("--embedding-models", nargs="+", choices=["bge", "minilm"], default=["bge", "minilm"])
    eval_parser.add_argument("--top-k", type=int, default=3)
    eval_parser.add_argument("--max-rows", type=int)
    eval_parser.add_argument("--use-process-guidance", action="store_true")
    eval_parser.add_argument("--llm-model", required=True)
    eval_parser.add_argument("--prompt-variant")
    eval_parser.add_argument("--reranker-model", choices=["bge"])
    eval_parser.add_argument("--rerank-pool-k", type=int, help="Initial retrieval size before reranking. Must be >= --top-k.")
    eval_parser.add_argument("--run-label", default="cli_evaluation")
    eval_parser.add_argument("--results-path", default="../data/evaluation_results/evaluation_runs.json")
    eval_parser.add_argument(
        "--keep-individual-predictions",
        action="store_true",
        help="Keep row-level predictions in saved reports. Default is summary metrics only.",
    )
    eval_parser.add_argument(
        "--earlyness-buckets",
        nargs="+",
        type=int,
        default=[5, 10, 20, 30],
        help="Bucket upper bounds for earlyness analysis, e.g. 5 10 20 30.",
    )
    eval_parser.set_defaults(func=eval_command)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
