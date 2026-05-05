import re
import json
import os
import time
from pathlib import Path

import requests
from dotenv import load_dotenv
load_dotenv()
REPO_ROOT = Path(__file__).resolve().parent.parent
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")


def _is_openai_detour_model(llm_model: str | None) -> bool:
    return llm_model in {
        "gpt-4o-mini",
        "gpt-4.1-mini",
        "gpt-4.1-mini-2025-04-14",
    }



def _split_prefix_events(prefix: str) -> list[str]:
    events = []
    current = []
    brace_depth = 0

    for char in prefix:
        if char == "{":
            brace_depth += 1
        elif char == "}":
            brace_depth = max(0, brace_depth - 1)

        if char == "," and brace_depth == 0:
            event = "".join(current).strip()
            if event:
                events.append(event)
            current = []
            continue

        current.append(char)

    final_event = "".join(current).strip()
    if final_event:
        events.append(final_event)

    return events


def _clean_event_label(event_text: str) -> str:
    event_text = event_text.split(" - Values:", 1)[0].strip()
    event_text = re.sub(r"^\d+\.\s*", "", event_text)
    return event_text


def _prefix_labels(prefix: str) -> list[str]:
    return [_clean_event_label(event) for event in _split_prefix_events(prefix)]


def _resolve_discovered_process_path(dataset_name: str | None) -> Path | None:
    if not dataset_name:
        return None

    variant_path = REPO_ROOT / "data" / dataset_name / "discovered_process.txt"
    if variant_path.exists():
        return variant_path

    legacy_discovery_dir = REPO_ROOT / "data" / "discovered_processes"
    if legacy_discovery_dir.exists():
        for path in legacy_discovery_dir.glob("*_discovered_process.txt"):
            log_name = path.stem.removesuffix("_discovered_process")
            if dataset_name.endswith(log_name) or log_name in dataset_name:
                return path

    return None


def _resolve_attribute_labels_path(dataset_name: str | None) -> Path | None:
    if not dataset_name:
        return None

    labels_path = REPO_ROOT / "data" / dataset_name / "attribute_labels.json"
    if labels_path.exists():
        return labels_path

    return None


def _extract_state_block(section_text: str, state: str) -> list[str]:
    lines = section_text.splitlines()
    target = f"- State: {state}"

    for index, line in enumerate(lines):
        if line == target:
            next_lines: list[str] = []
            for following_line in lines[index + 1:]:
                if following_line.startswith("- State: ") or not following_line.strip():
                    break
                if following_line.strip().startswith("Next:"):
                    next_lines.append(following_line.strip())
            return next_lines

    return []


def _extract_state_matches_from_file(discovered_path: Path, targets: dict[str, str]) -> dict[str, list[str]]:
    matches = {section: [] for section in targets}
    current_section = ""
    current_state = ""

    section_headers = {
        "Possible next steps by current activity:": "activity",
        "Possible next steps by last up to 3 activities:": "suffix",
        "Possible next steps by exact prefix state:": "exact",
    }

    with open(discovered_path, encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")
            stripped = line.strip()

            if line in section_headers:
                current_section = section_headers[line]
                current_state = ""
                continue

            if not stripped:
                continue

            if stripped.startswith("- State: "):
                current_state = stripped.removeprefix("- State: ")
                continue

            if (
                current_section in targets
                and current_state == targets[current_section]
                and stripped.startswith("Next:")
            ):
                matches[current_section].append(stripped)

    return matches


def build_process_guidance(prefix: str, dataset_name: str | None = None) -> str:
    discovered_path = _resolve_discovered_process_path(dataset_name)
    if discovered_path is None or not discovered_path.exists():
        return ""

    prefix_labels = _prefix_labels(prefix)
    if not prefix_labels:
        return ""

    exact_state = " | ".join(prefix_labels)
    suffix_state = " | ".join(prefix_labels[-3:])
    current_activity = prefix_labels[-1]

    state_matches = _extract_state_matches_from_file(
        discovered_path,
        targets={
            "exact": exact_state,
            "suffix": suffix_state,
            "activity": current_activity,
        },
    )
    exact_matches = state_matches["exact"]
    suffix_matches = state_matches["suffix"]
    activity_matches = state_matches["activity"]

    guidance_parts: list[str] = []
    if exact_matches:
        guidance_parts.append(
            "Observed next activities for the exact prefix state:\n"
            + "\n".join(f"- {line.replace('Next: ', '')}" for line in exact_matches)
        )
    if suffix_matches:
        guidance_parts.append(
            "Observed next activities for the last up to 3 activities of the prefix:\n"
            + "\n".join(f"- {line.replace('Next: ', '')}" for line in suffix_matches)
        )
    if activity_matches:
        guidance_parts.append(
            "Observed next activities for the current last activity of the prefix:\n"
            + "\n".join(f"- {line.replace('Next: ', '')}" for line in activity_matches)
        )

    if not guidance_parts:
        return ""

    return "\n\n".join(guidance_parts)


def build_attribute_legend(dataset_name: str | None = None) -> str:
    labels_path = _resolve_attribute_labels_path(dataset_name)
    if labels_path is None or not labels_path.exists():
        return ""

    with open(labels_path, encoding="utf-8") as f:
        labels = json.load(f)

    if not labels:
        return ""

    legend_lines = [f"- {abbr} = {label}" for abbr, label in labels.items()]
    return "# Attribute Labels\n" + "\n".join(legend_lines)


PROMPT_VARIANTS = {
    "prompt_1": {
        "identity": (
            "Act as a Process Mining assistant specialised in predictive process monitoring.\n"
            "Your task is to predict the next activity immediately after the last activity in the provided prefix.\n"
        ),
        "task_instructions": (
            "* Use the past traces and the last event values to reason.\n"
            "* Put your final answer only inside <answer></answer>, i.e., formatted as follows: \"<answer>predicted_activity_name</answer>\".\n"
            "* Return only <answer>predicted_activity</answer>."
        ),
    },
    "prompt_2": {
        "identity": (
            "Act as a Process Mining assistant specialised in predictive process monitoring.\n"
            "Your task is to predict the next activity immediately after the last activity of the provided prefix of an execution trace.\n"
            "Please reason step by step, and put your final answer within <answer></answer>, i.e., formatted as follows: '<answer>predicted_activity_name</answer>'."
        ),
        "task_instructions": (
            "* Decision Rule 1: Find the trace in the past traces that is equal or contains the sequence of activities of the prefix to predict.\n"
            "* Decision Rule 2: For the selected trace, identify the next activity and return it as the predicted next activity for that prefix.\n"
            "* Decision Rule 3: If multiple traces are equal or contain the sequence of activities of the prefix to predict, do not directly consider the most frequent next activity but choose as the next activity the one immediately following the last of the prefix in the matching trace, reasoning on the similarity of the values for the attributes in \"Values\" with the values for the attributes across the prefix.\n"
            "* Put your final answer only inside <answer></answer>, i.e., formatted as follows: \"<answer>predicted_activity_name</answer>\".\n"
            "* Return only <answer>predicted_activity</answer>."
        ),
    },
    "last_m_prompt": {
        "identity": (
            "Act as a Process Mining assistant specialised in predictive process monitoring.\n"
            "Your task is to predict the next activity immediately after the last activity in the provided prefix.\n"
        ),
        "task_instructions": (
            "* Use the past traces and the last event values to reason.\n"
            "* Use the last event values to assist the prediction of the sequence progression.\n"
            "* Put your final answer only inside <answer></answer>, i.e., formatted as follows: \"<answer>predicted_activity_name</answer>\".\n"
            "* Return only <answer>predicted_activity</answer>."
        ),
    },
}


def instructions_builder(
    past_traces: dict,
    process_guidance: str = "",
    attribute_legend: str = "",
    prompt_variant: str | None = None,
):
    prompt_config = PROMPT_VARIANTS.get(prompt_variant) if prompt_variant else None
    identity = ""
    task_instructions = ""
    if prompt_config is not None:
        identity = prompt_config["identity"]
        task_instructions = prompt_config["task_instructions"]
    
    traces_block = "\n".join(
        f'<past_trace id="{trace_id}">{trace_data.get("prefix", "")}</past_trace>\n'
        f'<next_activity id="{trace_id}">{trace_data.get("prediction", "")}</next_activity>'
        for trace_id, trace_data in past_traces.items()
    )
    process_guidance_block = ""
    if process_guidance:
        process_guidance_block = f"# Process Guidance\n{process_guidance}\n\n"
    attribute_legend_block = ""
    if attribute_legend:
        attribute_legend_block = f"{attribute_legend}\n\n"

    sections: list[str] = []
    if identity:
        sections.append(f"# Identity\n{identity}")
    if task_instructions:
        sections.append(f"# Instructions\n{task_instructions}")
    if attribute_legend_block:
        sections.append(attribute_legend_block.rstrip())
    if process_guidance_block:
        sections.append(process_guidance_block.rstrip())
    sections.append(f"# Past Traces\n{traces_block}")

    return "\n\n".join(sections)


def input_duilder(prefix: str):
    prefix = (f"# Prefix To Predict\n"
              f"<prefix>{prefix}</prefix>\n\n")
    return prefix


def api_call(
    past_traces: dict,
    prefix: str,
    dataset_name: str | None = None,
    use_process_guidance: bool = True,
    llm_model: str | None = None,
    prompt_variant: str | None = None,
):
    if not llm_model:
        raise ValueError("llm_model must be provided.")

    use_openai_detour = _is_openai_detour_model(llm_model)
    if use_openai_detour:
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY must be set for gpt-4o-mini calls.")
    elif not OLLAMA_BASE_URL:
        raise ValueError("OLLAMA_BASE_URL must be set in the environment.")

    process_guidance = ""
    if use_process_guidance:
        process_guidance = build_process_guidance(prefix, dataset_name=dataset_name)
    attribute_legend = build_attribute_legend(dataset_name=dataset_name)

    full_prompt = (
        instructions_builder(
            past_traces,
            process_guidance=process_guidance,
            attribute_legend=attribute_legend,
            prompt_variant=prompt_variant,
        )
        + "\n\n"
        + input_duilder(prefix)
    )
    openai_prompt_variant = prompt_variant if prompt_variant in PROMPT_VARIANTS else "prompt_1"
    openai_prompt_config = PROMPT_VARIANTS[openai_prompt_variant]
    identity = openai_prompt_config.get("identity", "")
    task_instructions = openai_prompt_config.get("task_instructions", "")
    developer_sections: list[str] = []
    if identity:
        developer_sections.append(f"# Identity\n{identity}")
    if task_instructions:
        developer_sections.append(f"# Instructions\n{task_instructions}")
    developer_prompt = "\n\n".join(developer_sections)

    user_prompt = (
        instructions_builder(
            past_traces,
            process_guidance=process_guidance,
            attribute_legend=attribute_legend,
            prompt_variant=None,
        )
        + "\n\n"
        + input_duilder(prefix)
    )

    max_retries = 3
    retry_wait_seconds = 20
    request_timeout_seconds = 120

    for attempt in range(max_retries + 1):
        provider_name = "OpenAI" if use_openai_detour else "Ollama"
        try:
            if use_openai_detour:
                messages = []
                messages.append({"role": "developer", "content": developer_prompt})
                messages.append({"role": "user", "content": user_prompt})

                response = requests.post(
                    f"{OPENAI_BASE_URL.rstrip('/')}/chat/completions",
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {OPENAI_API_KEY}",
                    },
                    json={
                        "model": llm_model,
                        "messages": messages,
                        "temperature": 0.0,
                    },
                    timeout=request_timeout_seconds,
                )
            else:
                response = requests.post(
                    f"{OLLAMA_BASE_URL.rstrip('/')}/api/generate",
                    headers={"Content-Type": "application/json"},
                    data=json.dumps(
                        {
                            "model": llm_model,
                            "prompt": full_prompt,
                            "stream": False,
                            "options": {"temperature": 0.0},
                            "keep_alive": "30m"
                        }
                    ),
                    timeout=request_timeout_seconds,
                )
            response.raise_for_status()
            print("waiting...")
            body = response.json()
            if use_openai_detour:
                return body["choices"][0]["message"]["content"]
            return body["response"]
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError, requests.exceptions.HTTPError) as exc:
            if attempt == max_retries:
                print(
                    f"{provider_name} call failed after {max_retries + 1} attempts: {exc}. "
                    "Falling back to None prediction."
                )
                return ""
            print(
                f"{provider_name} call failed (attempt {attempt + 1}/{max_retries + 1}): {exc}. "
                f"Retrying in {retry_wait_seconds} seconds..."
            )
            time.sleep(retry_wait_seconds)


def stop_ollama_model(llm_model: str) -> None:
    if _is_openai_detour_model(llm_model):
        return

    if not OLLAMA_BASE_URL or not llm_model:
        return

    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL.rstrip('/')}/api/generate",
            headers={"Content-Type": "application/json"},
            data=json.dumps(
                {
                    "model": llm_model,
                    "prompt": "",
                    "stream": False,
                    "keep_alive": 0,
                }
            ),
            timeout=30,
        )
        response.raise_for_status()
    except requests.exceptions.RequestException as exc:
        print(f"Warning: failed to stop model '{llm_model}': {exc}")
