import pandas as pd
import pm4py
import csv
from pathlib import Path

def read_clean_log(
    filename: str | Path, 
    trace_identifier: str = "case:concept:name"
) -> pd.DataFrame:

    filename_str = str(filename)
    
    xes_log  = pm4py.read_xes(filename_str)
    df_log = pm4py.convert_to_dataframe(xes_log)

    # keep only complete events if life cycle column is present
    allowed = {"complete", "completed"}
    if "lifecycle:transition" in df_log.columns:
        if "2013" not in filename_str:
            df_log = df_log[
                    df_log["lifecycle:transition"].astype(str).str.strip().str.lower().isin(allowed)
                ].reset_index(drop=True)
        else:
            df_log = df_log.reset_index(drop=True)

    # ignore trace attributes except trace id, keep event attributes
    keep_cols = [
            c for c in df_log.columns if not c.startswith("case:") or c == trace_identifier
        ]

    df_log = df_log[keep_cols].copy()
    
    return df_log

def keep_last_m_values(activity_prefix: str, m: int) -> str:
    token = " - Values: {"
    spans = []
    pos = 0

    while True:
        start = activity_prefix.find(token, pos)
        if start == -1:
            break

        brace_start = activity_prefix.find("{", start)
        depth = 0
        end = None

        for i in range(brace_start, len(activity_prefix)):
            if activity_prefix[i] == "{":
                depth += 1
            elif activity_prefix[i] == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break

        if end is None:
            break

        spans.append((start, end))
        pos = end

    if len(spans) <= m:
        return activity_prefix

    parts = []
    last = 0

    for start, end in spans[:-m]:
        parts.append(activity_prefix[last:start])
        last = end

    parts.append(activity_prefix[last:])
    return "".join(parts)

def build_prefixes(
    output_path: str | Path,
    df_log: pd.DataFrame,
    base: int = 1,
    gap: int = 3,
    trace_identifier: str = "case:concept:name",
    seen_prefixes: set[str] | None = None,
    m: int = 1
) -> None:
    """
    This function recieves a Dataframe, creates the prefixes, and adds them to a csv. 
    """
    
    is_bpic_2013 = "2013" in str(output_path)
        
    if seen_prefixes is None:
        seen_prefixes = set()
    prefixes = []
    j_map = {}

    # iterate through events of a single trace
    for _, df_trace in df_log.groupby(trace_identifier, sort=False):

        df_trace = df_trace.sort_values(["time:timestamp"]).reset_index(drop=True)

        if len(df_trace) <= 2:
            continue
        
        activity_prefix = ""
        deduplication_list = ""
        values = {}

        # keep only indexes that match the selected bucketing
        for i in range(base, len(df_trace)-1, gap):

            if i == base:
                start = 0
            else:
                start = i-gap+1

            # process events from last to next gap
            for event_index in range(start, i+1):
                
                event = df_trace.iloc[event_index]

                for key, value in event.items():
        
                    if not is_bpic_2013:
                        if key in {"concept:name", trace_identifier}:
                            continue
                    else:
                        if key in {"concept:name", trace_identifier, "lifecycle:transition", "org:resource", "org:role"}:
                            continue

                    if key not in j_map:
                        j_map[key] = ''.join([part[:2] for part in key.split(':')])
                    values[j_map[key]] = str(value)
                
                keep_values = (i - event_index) < m
                event_attr = f" - Values: {values}" if keep_values else ""
                    
                if not is_bpic_2013:
                    activity_prefix += ("," if activity_prefix else "") + f"{event['concept:name']}{event_attr}"
                    deduplication_list += f"{event['concept:name']}"
                else:
                    activity_prefix += (("," if activity_prefix else "") + f"{event['concept:name']}+{event['lifecycle:transition']}+{event['org:resource']}+{event['org:role']}{event_attr}")
                    deduplication_list += f"{event['concept:name']}+{event['lifecycle:transition']}+{event['org:resource']}+{event['org:role']}"
                    
            activity_prefix = keep_last_m_values(activity_prefix=activity_prefix, m=m)
            
            if deduplication_list in seen_prefixes:
                continue

            seen_prefixes.add(deduplication_list)

            values = dict(sorted(values.items()))
            

            if not is_bpic_2013:
                final_prefix_string = f"{activity_prefix} - {df_trace.iloc[i+1]['concept:name']}"
            else:
                final_prefix_string = f"{activity_prefix} - {df_trace.iloc[i+1]['concept:name']}+{df_trace.iloc[i+1]['lifecycle:transition']}+{df_trace.iloc[i+1]['org:resource']}+{df_trace.iloc[i+1]['org:role']}"
            prefixes.append(final_prefix_string)

    if prefixes:
        open_mode = "a" if Path(output_path).exists() else "w"
        convert_to_csv(prefix_list=prefixes, output_path=output_path, open_mode=open_mode)
        
    print(f"{len(prefixes)} prefixes were built successfully")


def convert_to_csv(
    prefix_list: list[str], 
    output_path: str | Path, 
    open_mode: str = "w"
) -> None:
    
    rows = []
    for row in prefix_list:
        left, right = row.split(" - Values: ", 1)
        values_part, prediction = right.rsplit(" - ", 1)
        rows.append((f"{left} - Values: {values_part}".strip(), prediction.strip()))
        
    with open(output_path, open_mode, newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if open_mode == "w":
                w.writerow(["prefix", "prediction"])
            w.writerows(rows)


def generate_test_set(
    csv_prefixes: str | Path, 
    test_set_proportion: float = 0.3
) -> None:
    
    csv_prefixes = Path(csv_prefixes)
    df = pd.read_csv(csv_prefixes)
    test_set_size = int(len(df) * test_set_proportion)
    
    test_df = df.sample(n=test_set_size, random_state=42)
    train_df = df.drop(test_df.index)
    
    test_csv_prefixes = (
            Path(__file__).resolve().parent.parent
            / "data" 
            / "test_csv_prefixes" 
            / csv_prefixes.name
    )
    test_csv_prefixes.parent.mkdir(parents=True, exist_ok=True)
    
    test_df.to_csv(test_csv_prefixes, index=False)
    train_df.to_csv(csv_prefixes, index=False)


def process_log(
    dataset_xes: str | Path,
    shard_output_dir: str | Path | None = None,
    base: int = 1,
    gap: int = 3,
    trace_identifier: str = "case:concept:name",
    test_set_proportion: float = 0.3,
    m: int = 1
) -> Path:
    """
    This is a wrapper that takes an xes file and produces the retrieval csv prefixes and test csv prefixes files.
    """
    
    dataset_xes = Path(dataset_xes)
    base_stem = dataset_xes.stem

    num_shards = 1
    
    dataset_size = dataset_xes.stat().st_size / (1024 * 1024)
    
    if dataset_size > 300:
        print(f"Dataset size is {dataset_size} mb. Initiating sharding...")
        if shard_output_dir is None:
            shard_output_dir = (
                Path(__file__).resolve().parent.parent
                / "data"
                / "xes_logs"
                / f"{base_stem}_Shards"
            )
        num_shards = int(dataset_size // 100) + 2
        print(f"{num_shards} shards will be at {shard_output_dir}.")
        
    is_full_file = test_set_proportion <= 0
    output_dir_name = "full_csv_prefixes" if is_full_file else "retrieval_csv_prefixes"
    last_m_id = f"last_{m}_" if m > 1 else ""
    
    file_name = (
        f"full_{last_m_id}b{base}_g{gap}_{base_stem}.csv"
        if is_full_file
        else f"{last_m_id}b{base}_g{gap}_{base_stem}.csv"
    )

    output_csv_path = (
        Path(__file__).resolve().parent.parent 
        / "data"
        / output_dir_name
        / file_name
    )
    
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    if output_csv_path.exists():
        output_csv_path.unlink()
        
    if num_shards != 1:
        shard_paths = shard_log(
            input_xes_path=dataset_xes,
            num_shards=num_shards,
            output_dir=shard_output_dir,
        )
    else:
        shard_paths = [dataset_xes]

    global_seen_prefixes: set[str] = set()
    for shard_path in shard_paths:
        df_log = read_clean_log(shard_path, trace_identifier=trace_identifier)
        build_prefixes(
            output_path=output_csv_path,
            df_log=df_log,
            trace_identifier=trace_identifier,
            base=base,
            gap=gap,
            seen_prefixes=global_seen_prefixes,
            m=m
        )

    if test_set_proportion > 0:
        generate_test_set(output_csv_path, test_set_proportion)
    
    return output_csv_path


def shard_log(
    input_xes_path: str | Path,
    num_shards: int,
    output_dir: str | Path | None = None,
) -> list[Path]:
    """
    Split an XES log into `num_shards` valid XES files with near-equal trace counts.

    Output file names follow: <input_stem>_1.xes, <input_stem>_2.xes, ... <input_stem>_N.xes
    """
    if num_shards <= 0:
        raise ValueError("num_shards must be > 0")

    input_xes_path = Path(input_xes_path)
    if not input_xes_path.exists():
        raise FileNotFoundError(f"Input XES not found: {input_xes_path}")

    out_dir = Path(output_dir) if output_dir else input_xes_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    trace_open = b"<trace>"
    trace_close = b"</trace>"
    log_close = b"</log>\n"

    def _read_header_and_count_traces(path: Path, block_size: int = 1024 * 1024) -> tuple[bytes, int]:
        header = bytearray()
        found_first_trace = False
        trace_count = 0
        prev_tail = b""

        with open(path, "rb") as f:
            while True:
                chunk = f.read(block_size)
                if not chunk:
                    break
                data = prev_tail + chunk

                if not found_first_trace:
                    idx = data.find(trace_open)
                    if idx != -1:
                        found_first_trace = True
                        header.extend(data[:idx])
                    else:
                        # Keep only a short overlap in case "<trace>" spans chunks.
                        keep = len(trace_open) - 1
                        if len(data) > keep:
                            header.extend(data[:-keep])
                            prev_tail = data[-keep:]
                        else:
                            prev_tail = data
                        continue

                trace_count += data.count(trace_open)
                prev_tail = data[-(len(trace_open) - 1):] if len(data) >= (len(trace_open) - 1) else data

        if not found_first_trace:
            raise ValueError(f"No <trace> elements found in: {path}")
        return bytes(header), trace_count

    header_bytes, total_traces = _read_header_and_count_traces(input_xes_path)
    q, r = divmod(total_traces, num_shards)
    shard_sizes = [q + 1 if i < r else q for i in range(num_shards)]

    shard_paths: list[Path] = [out_dir / f"{input_xes_path.stem}_{i}.xes" for i in range(1, num_shards + 1)]

    # Initialize all shard files with header so each output remains valid XES.
    for p in shard_paths:
        with open(p, "wb") as out:
            out.write(header_bytes)

    current_shard_idx = 0
    traces_in_current_shard = 0
    buffer = b""
    started = False

    with open(input_xes_path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            buffer += chunk

            while True:
                if not started:
                    first_trace_idx = buffer.find(trace_open)
                    if first_trace_idx == -1:
                        # Keep small tail for split tag detection and continue reading.
                        keep = len(trace_open) - 1
                        buffer = buffer[-keep:] if len(buffer) > keep else buffer
                        break
                    buffer = buffer[first_trace_idx:]
                    started = True

                start_idx = buffer.find(trace_open)
                if start_idx == -1:
                    break
                end_idx = buffer.find(trace_close, start_idx)
                if end_idx == -1:
                    # Keep from start_idx onward because trace may be incomplete.
                    buffer = buffer[start_idx:]
                    break

                trace_block = buffer[start_idx:end_idx + len(trace_close)]
                if current_shard_idx >= len(shard_paths):
                    # Safety guard; should not happen if counts are consistent.
                    break

                with open(shard_paths[current_shard_idx], "ab") as out:
                    out.write(trace_block)

                traces_in_current_shard += 1
                buffer = buffer[end_idx + len(trace_close):]

                if traces_in_current_shard == shard_sizes[current_shard_idx]:
                    current_shard_idx += 1
                    traces_in_current_shard = 0
                    if current_shard_idx >= len(shard_paths):
                        break

            if current_shard_idx >= len(shard_paths):
                break

    # Close all shards with </log>.
    for p in shard_paths:
        with open(p, "ab") as out:
            out.write(log_close)

    return shard_paths
