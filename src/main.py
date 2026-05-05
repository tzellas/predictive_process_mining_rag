from pathlib import Path

from event_log_preprocessing import process_log
from process_discovery import discover_log_for_variant
from retrieval import Retrieval
from utils import build_attribute_labels_json


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    base = 1
    gap = 3
    m = 3

    dataset_xes_paths = [
        repo_root / "data" / "xes_logs" / "BPI_Challenge_2012.xes",
        repo_root / "data" / "xes_logs" / "BPI_Challenge_2019.xes",
        repo_root / "data" / "xes_logs" / "BPI_Challenge_2020_InternationalDeclarations.xes",
    ]

    for dataset_xes_path in dataset_xes_paths:
        print(f"Processing {dataset_xes_path.name} with b={base}, g={gap}, m={m}")
        retrieval_csv_path, test_csv_path, train_xes_path, _ = process_log(
            dataset_xes=dataset_xes_path,
            base=base,
            gap=gap,
            m=m,
        )

        discovered_process_path = discover_log_for_variant(
            train_xes_path=train_xes_path,
            original_xes_path=dataset_xes_path,
            base=base,
            gap=gap,
            m=m,
        )

        dataset_dir = retrieval_csv_path.parent
        labels_path = build_attribute_labels_json(dataset_dir)

        for model_id in ("bge", "minilm"):
            print(f"Building collection for {dataset_dir.name} with {model_id}")
            retrieval = Retrieval(
                dataset=retrieval_csv_path,
                test_set=test_csv_path,
                model_id=model_id,
            )
            retrieval.store_embeddings()

        print(f"retrieval.csv: {retrieval_csv_path}")
        print(f"test.csv: {test_csv_path}")
        print(f"discovered_process.txt: {discovered_process_path}")
        print(f"attribute_labels.json: {labels_path}")
        print()

    # Previous evaluation main:
    # top_k = 5
    # max_rows = 300
    # use_process_guidance = False
    # llm_model = "gpt-4o-mini"
    # prompt_variant = "nano_4o_mini"
    #
    # dataset_dirs = [
    #     repo_root / "data" / "b1_g3_BPI_Challenge_2012",
    #     repo_root / "data" / "b1_g3_BPI_Challenge_2019",
    #     repo_root / "data" / "b1_g3_BPI_Challenge_2020_InternationalDeclarations",
    # ]
    #
    # retrieval_setups = []
    # for dataset_dir in dataset_dirs:
    #     retrieval_csv_path = dataset_dir / "retrieval.csv"
    #     test_csv_path = dataset_dir / "test.csv"
    #     for model_id in ("bge", "minilm"):
    #         retrieval_setups.append(
    #             Retrieval(
    #                 dataset=retrieval_csv_path,
    #                 test_set=test_csv_path,
    #                 model_id=model_id,
    #             )
    #         )
    #
    # evaluation_report = evaluate_multiple_datasets(
    #     retrieval_setups,
    #     top_k=top_k,
    #     max_rows=max_rows,
    #     use_process_guidance=use_process_guidance,
    #     llm_model=llm_model,
    #     prompt_variant=prompt_variant,
    # )
    #
    # run_label = "baseline_without_process_guidance"
    # print_reports(run_label, evaluation_report["dataset_reports"])
    #
    # evaluation_runs = [
    #     {
    #         "timestamp": datetime.now().isoformat(),
    #         "run_label": run_label,
    #         "datasets": [report["dataset"] for report in evaluation_report["dataset_reports"]],
    #         "top_k": top_k,
    #         "max_rows": max_rows,
    #         "use_process_guidance": use_process_guidance,
    #         "llm": llm_model,
    #         "prompt_variant": prompt_variant,
    #         "reports": evaluation_report["dataset_reports"],
    #     }
    # ]
    #
    # results_path = dump_evaluation_results(evaluation_runs)
    # print(f"Saved evaluation results to {results_path}")


if __name__ == "__main__":
    main()
