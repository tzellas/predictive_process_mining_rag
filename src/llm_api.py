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

def _resolve_attribute_labels_path(dataset_name: str | None) -> Path | None:
    if not dataset_name:
        return None

    labels_path = REPO_ROOT / "data" / dataset_name / "attribute_labels.json"
    if labels_path.exists():
        return labels_path

    return None




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

    attribute_legend = build_attribute_legend(dataset_name=dataset_name)

    full_prompt = (
        instructions_builder(
            past_traces,
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
