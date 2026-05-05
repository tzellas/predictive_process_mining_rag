from collections import Counter, defaultdict
from pathlib import Path

from event_log_preprocessing import read_clean_log
from event_log_preprocessing import variant_dir_path


TRACE_IDENTIFIER = "case:concept:name"
REPO_ROOT = Path(__file__).resolve().parent.parent


def build_event_label(event: dict, is_bpic_2013: bool) -> str:
    if is_bpic_2013:
        return (
            f"{event['concept:name']}+"
            f"{event['lifecycle:transition']}+"
            f"{event['org:resource']}+"
            f"{event['org:role']}"
        )
    return str(event["concept:name"])


def _join_state(labels: list[str]) -> str:
    return " | ".join(labels)


def discover_prefix_state_map(xes_path: str | Path) -> dict:
    xes_path = Path(xes_path)
    df_log = read_clean_log(xes_path)
    is_bpic_2013 = "2013" in xes_path.stem

    start_activities = Counter()
    end_activities = Counter()
    direct_follows = Counter()
    activity_next_steps = defaultdict(Counter)
    prefix_state_next_steps = defaultdict(Counter)
    suffix_3_next_steps = defaultdict(Counter)

    total_traces = 0
    all_activities: set[str] = set()

    for _, df_trace in df_log.groupby(TRACE_IDENTIFIER, sort=False):
        df_trace = df_trace.reset_index(drop=True)
        df_trace["_original_order"] = range(len(df_trace))
        df_trace = df_trace.sort_values(
            ["time:timestamp", "_original_order"],
            kind="stable",
        ).reset_index(drop=True)
        df_trace = df_trace.drop(columns="_original_order")
        records = df_trace.to_dict("records")
        if not records:
            continue

        labels = [build_event_label(event, is_bpic_2013) for event in records]
        total_traces += 1
        all_activities.update(labels)

        start_activities[labels[0]] += 1
        end_activities[labels[-1]] += 1

        for source, target in zip(labels, labels[1:]):
            direct_follows[(source, target)] += 1
            activity_next_steps[source][target] += 1

        for i in range(1, len(labels)):
            prefix_key = _join_state(labels[:i])
            next_activity = labels[i]
            prefix_state_next_steps[prefix_key][next_activity] += 1

            suffix_key = _join_state(labels[max(0, i - 3):i])
            suffix_3_next_steps[suffix_key][next_activity] += 1

    return {
        "log_name": xes_path.stem,
        "total_traces": total_traces,
        "total_events": len(df_log),
        "unique_activities": len(all_activities),
        "start_activities": start_activities,
        "end_activities": end_activities,
        "direct_follows": direct_follows,
        "activity_next_steps": activity_next_steps,
        "prefix_state_next_steps": prefix_state_next_steps,
        "suffix_3_next_steps": suffix_3_next_steps,
    }


def _format_counter(counter: Counter) -> list[str]:
    return [f"- {label} ({count})" for label, count in counter.most_common()]


def _format_state_map(title: str, state_map: dict[str, Counter]) -> list[str]:
    lines = ["", title]
    for state in sorted(state_map):
        lines.append(f"- State: {state}")
        for next_activity, count in state_map[state].most_common():
            lines.append(f"  Next: {next_activity} ({count})")
    return lines


def format_discovery_text(discovery: dict) -> str:
    lines: list[str] = [
        f"Discovered process summary for: {discovery['log_name']}",
        f"Total traces: {discovery['total_traces']}",
        f"Total events: {discovery['total_events']}",
        f"Unique activities: {discovery['unique_activities']}",
        "",
        "Start activities:",
        *_format_counter(discovery["start_activities"]),
        "",
        "End activities:",
        *_format_counter(discovery["end_activities"]),
        "",
        "Directly-follows transitions:",
    ]

    for (source, target), count in discovery["direct_follows"].most_common():
        lines.append(f"- {source} -> {target} ({count})")

    lines.extend(
        _format_state_map(
            "Possible next steps by current activity:",
            discovery["activity_next_steps"],
        )
    )
    lines.extend(
        _format_state_map(
            "Possible next steps by last up to 3 activities:",
            discovery["suffix_3_next_steps"],
        )
    )
    lines.extend(
        _format_state_map(
            "Possible next steps by exact prefix state:",
            discovery["prefix_state_next_steps"],
        )
    )

    return "\n".join(lines) + "\n"


def discover_log(
    xes_path: str | Path,
    output_dir: str | Path = "data",
    output_name: str | None = None,
) -> Path:
    xes_path = Path(xes_path)
    output_dir = REPO_ROOT / Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    discovery = discover_prefix_state_map(xes_path)
    file_name = f"{output_name}.txt" if output_name is not None else f"{xes_path.stem}_discovered_process.txt"
    output_path = output_dir / file_name
    output_path.write_text(format_discovery_text(discovery), encoding="utf-8")
    return output_path


def discover_log_for_variant(
    train_xes_path: str | Path,
    original_xes_path: str | Path,
    base: int,
    gap: int,
    m: int,
    split_mode: str = "trace",
) -> Path:
    output_dir = variant_dir_path(
        dataset_xes=original_xes_path,
        base=base,
        gap=gap,
        m=m,
        split_mode=split_mode,
    ).relative_to(REPO_ROOT)
    return discover_log(
        xes_path=train_xes_path,
        output_dir=output_dir,
        output_name="discovered_process",
    )


def main() -> None:
    raise RuntimeError(
        "Use discover_log(...) from the pipeline with a specific train XES path."
    )


if __name__ == "__main__":
    main()
