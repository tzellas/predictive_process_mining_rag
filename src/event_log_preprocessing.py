import pandas as pd
import pm4py
import csv
import random
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

def keep_last_m_values(
    activity_prefix: str, 
    m: int
) -> str:
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
    seen_prefixes: set[tuple[str, str]] | None = None,
    m: int = 1,
) -> None:
    """
    This function recieves a Dataframe, creates the prefixes, and adds them to a csv. 
    """
    
    is_bpic_2013 = "2013" in str(output_path)
        
    if seen_prefixes is None:
        seen_prefixes = set()
    prefixes: list[tuple[str, str]] = []
    j_map = {}

    # iterate through events of a single trace
    for _, df_trace in df_log.groupby(trace_identifier, sort=False):

        df_trace = df_trace.reset_index(drop=True)
        df_trace["_original_order"] = range(len(df_trace))
        df_trace = df_trace.sort_values(
            ["time:timestamp", "_original_order"],
            kind="stable",
        ).reset_index(drop=True)
        df_trace = df_trace.drop(columns="_original_order")
        records = df_trace.to_dict("records")

        if len(records) <= 2:
            continue
        
        activity_prefix = ""
        deduplication_list = ""
        values = {}

        # keep only indexes that match the selected bucketing
        for i in range(base, len(records)-1, gap):

            if i == base:
                start = 0
            else:
                start = i-gap+1

            # process events from last to next gap
            for event_index in range(start, i+1):
                
                event = records[event_index]

                for key, value in event.items():
        
                    if not is_bpic_2013:
                        if key in {"concept:name", trace_identifier}:
                            continue
                    else:
                        if key in {"concept:name", trace_identifier, "lifecycle:transition", "org:resource", "org:role"}:
                            continue

                    if key not in j_map:
                        j_map[key] = ''.join([part[:2] for part in key.replace(" ", ":").split(':')])
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
            
            values = dict(sorted(values.items()))
            
            if not is_bpic_2013:
                prediction = f"{records[i+1]['concept:name']}"
            else:
                prediction = (
                    f"{records[i+1]['concept:name']}+"
                    f"{records[i+1]['lifecycle:transition']}+"
                    f"{records[i+1]['org:resource']}+"
                    f"{records[i+1]['org:role']}"
                )

            dedup_key = (deduplication_list, prediction)

            if dedup_key in seen_prefixes:
                continue

            seen_prefixes.add(dedup_key)
            prefixes.append((activity_prefix, prediction))

    if prefixes:
        open_mode = "a" if Path(output_path).exists() else "w"
        convert_to_csv(prefix_list=prefixes, output_path=output_path, open_mode=open_mode)
        
    print(f"{len(prefixes)} prefixes were built successfully")


def convert_to_csv(
    prefix_list: list[tuple[str, str]], 
    output_path: str | Path, 
    open_mode: str = "w"
) -> None:

    with open(output_path, open_mode, newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if open_mode == "w":
                w.writerow(["prefix", "prediction"])
            w.writerows(prefix_list)


def _variant_file_name(
    base_stem: str,
    base: int,
    gap: int,
    m: int,
) -> str:
    last_m_id = f"last_{m}_" if m > 1 else ""
    return f"{last_m_id}b{base}_g{gap}_{base_stem}.csv"


def _variant_dir_name(
    base_stem: str,
    base: int,
    gap: int,
    m: int,
    split_mode: str = "trace",
) -> str:
    variant_name = _variant_file_name(
        base_stem=base_stem,
        base=base,
        gap=gap,
        m=m,
    ).removesuffix(".csv")
    if split_mode == "row":
        return f"{variant_name}_row"
    return variant_name


def split_xes_dir(input_xes_path: str | Path) -> Path:
    input_xes_path = Path(input_xes_path)
    return (
        Path(__file__).resolve().parent.parent
        / "data"
        / "xes_logs"
        / f"{input_xes_path.stem}_split"
    )


def split_xes_paths(input_xes_path: str | Path) -> tuple[Path, Path]:
    input_xes_path = Path(input_xes_path)
    out_dir = split_xes_dir(input_xes_path)
    return (
        out_dir / f"{input_xes_path.stem}_train.xes",
        out_dir / f"{input_xes_path.stem}_test.xes",
    )


def variant_dir_path(
    dataset_xes: str | Path,
    base: int,
    gap: int,
    m: int,
    split_mode: str = "trace",
) -> Path:
    dataset_xes = Path(dataset_xes)
    return (
        Path(__file__).resolve().parent.parent
        / "data"
        / _variant_dir_name(
            base_stem=dataset_xes.stem,
            base=base,
            gap=gap,
            m=m,
            split_mode=split_mode,
        )
    )


def split_xes_train_test(
    input_xes_path: str | Path,
    test_set_proportion: float = 0.3,
    output_dir: str | Path | None = None,
    random_state: int = 42,
) -> tuple[Path, Path]:
    input_xes_path = Path(input_xes_path)
    if not input_xes_path.exists():
        raise FileNotFoundError(f"Input XES not found: {input_xes_path}")

    out_dir = Path(output_dir) if output_dir is not None else split_xes_dir(input_xes_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path, test_path = split_xes_paths(input_xes_path)

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
    test_set_size = int(total_traces * test_set_proportion)
    rng = random.Random(random_state)
    test_indices = set(rng.sample(range(total_traces), test_set_size))

    for output_path in (train_path, test_path):
        with open(output_path, "wb") as out:
            out.write(header_bytes)

    buffer = b""
    started = False
    trace_index = 0

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
                    buffer = buffer[start_idx:]
                    break

                trace_block = buffer[start_idx:end_idx + len(trace_close)]
                destination = test_path if trace_index in test_indices else train_path

                with open(destination, "ab") as out:
                    out.write(trace_block)

                trace_index += 1
                buffer = buffer[end_idx + len(trace_close):]

    for output_path in (train_path, test_path):
        with open(output_path, "ab") as out:
            out.write(log_close)

    return train_path, test_path


def _process_single_log_to_csv(
    dataset_xes: str | Path,
    shard_output_dir: str | Path | None = None,
    base: int = 1,
    gap: int = 3,
    trace_identifier: str = "case:concept:name",
    m: int = 1,
    split_mode: str = "trace",
) -> Path:
    dataset_xes = Path(dataset_xes)
    base_stem = dataset_xes.stem

    num_shards = 1
    dataset_size = dataset_xes.stat().st_size / (1024 * 1024)

    if dataset_size > 300:
        print(f"Dataset size is {dataset_size} mb. Initiating sharding...")
        shard_output_dir = (
            Path(__file__).resolve().parent.parent
            / "data"
            / "xes_logs"
            / f"{base_stem}_Shards"
        )
        num_shards = int(dataset_size // 100) + 2
        print(f"{num_shards} shards will be at {shard_output_dir}.")

    variant_dir_name = _variant_dir_name(
        base_stem=base_stem.removesuffix("_train").removesuffix("_test"),
        base=base,
        gap=gap,
        m=m,
        split_mode=split_mode,
    )
    output_csv_path = (
        Path(__file__).resolve().parent.parent
        / "data"
        / variant_dir_name
        / f"{base_stem}.csv"
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

    global_seen_prefixes: set[tuple[str, str]] = set()
    for shard_path in shard_paths:
        df_log = read_clean_log(shard_path, trace_identifier=trace_identifier)
        build_prefixes(
            output_path=output_csv_path,
            df_log=df_log,
            trace_identifier=trace_identifier,
            base=base,
            gap=gap,
            seen_prefixes=global_seen_prefixes,
            m=m,
        )

    return output_csv_path


def process_log(
    dataset_xes: str | Path,
    shard_output_dir: str | Path | None = None,
    base: int = 1,
    gap: int = 3,
    trace_identifier: str = "case:concept:name",
    test_set_proportion: float = 0.3,
    m: int = 1,
    split_mode: str = "trace",
) -> tuple[Path, Path, Path, Path]:
    dataset_xes = Path(dataset_xes)
    if split_mode not in {"trace", "row"}:
        raise ValueError(f"Unsupported split_mode: {split_mode}")

    variant_dir = variant_dir_path(
        dataset_xes=dataset_xes,
        base=base,
        gap=gap,
        m=m,
        split_mode=split_mode,
    )
    retrieval_csv_path = variant_dir / "retrieval.csv"
    test_csv_path = variant_dir / "test.csv"
    variant_dir.mkdir(parents=True, exist_ok=True)

    if retrieval_csv_path.exists():
        retrieval_csv_path.unlink()
    if test_csv_path.exists():
        test_csv_path.unlink()

    if split_mode == "trace":
        train_xes_path, test_xes_path = split_xes_train_test(
            dataset_xes,
            test_set_proportion=test_set_proportion,
        )

        train_csv_tmp = _process_single_log_to_csv(
            dataset_xes=train_xes_path,
            shard_output_dir=shard_output_dir,
            base=base,
            gap=gap,
            trace_identifier=trace_identifier,
            m=m,
            split_mode=split_mode,
        )
        test_csv_tmp = _process_single_log_to_csv(
            dataset_xes=test_xes_path,
            shard_output_dir=shard_output_dir,
            base=base,
            gap=gap,
            trace_identifier=trace_identifier,
            m=m,
            split_mode=split_mode,
        )

        train_csv_tmp.replace(retrieval_csv_path)
        test_csv_tmp.replace(test_csv_path)
        return retrieval_csv_path, test_csv_path, train_xes_path, test_xes_path

    full_csv_tmp = _process_single_log_to_csv(
        dataset_xes=dataset_xes,
        shard_output_dir=shard_output_dir,
        base=base,
        gap=gap,
        trace_identifier=trace_identifier,
        m=m,
        split_mode=split_mode,
    )

    df = pd.read_csv(full_csv_tmp)
    if df.empty:
        raise ValueError(f"No prefix rows generated for: {dataset_xes}")

    test_size = int(len(df) * test_set_proportion)
    test_size = min(test_size, len(df))
    test_df = df.sample(n=test_size, random_state=42).copy()
    retrieval_df = df.drop(index=test_df.index).copy()

    retrieval_df.to_csv(retrieval_csv_path, index=False)
    test_df.to_csv(test_csv_path, index=False)
    full_csv_tmp.unlink(missing_ok=True)

    return retrieval_csv_path, test_csv_path, None, None


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
