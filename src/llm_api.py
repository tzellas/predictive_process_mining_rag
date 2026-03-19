from dotenv import load_dotenv
from openai import OpenAI
load_dotenv()
client = OpenAI()

def instructions_builder(past_traces: dict):
    
    identity = (
        "Act as a Process Mining assistant specialised in predictive process monitoring.\n"
        "Your task is to predict the next activity immediately after the last activity of the provided prefix of an execution trace.\n"
        "Please reason step by step."
    )

    task_instructions = (
        "* Find the trace in the past traces that is equal or contains the sequence of activities of the prefix to predict.\n"
        "* For the selected trace, identify the next activity and return it as the predicted next activity for that prefix.\n"
        "* If multiple traces are equal or contain the sequence of activities of the prefix to predict, do not directly consider the most frequent next activity but choose as the next activity the one immediately following the last of the prefix in the matching trace reasoning on the similarity of its values for the attributes in \"Values\" with the values for the attributes of the prefix.\n"
        "* Put your final answer within \\boxed{}, i.e., formatted as follows: \"\\boxed{predicted_activity_name}\"."
    )
    
    traces_block = "\n".join(
        f'<past_trace id="{trace_id}">{trace_data.get("prefix", "")}</past_trace>\n'
        f'<next_activity id="{trace_id}">{trace_data.get("prediction", "")}</next_activity>'
        for trace_id, trace_data in past_traces.items()
    )
    
    instructions = (
        f"# Identity\n"
        f"{identity}\n\n"
        f"# Instructions\n"
        f"{task_instructions}\n\n"
        f"# Past Traces\n"
        f"{traces_block}"
    )
    
    return instructions 


def input_duilder(prefix: str):
    prefix = (f"# Prefix To Predict\n"
              f"<prefix>{prefix}</prefix>\n\n")
    return prefix


def api_call(past_traces: dict, prefix: str):
    response = client.responses.create(
        model="gpt-5.2",
        instructions=instructions_builder(past_traces),
        input=input_duilder(prefix),
        reasoning={"effort": "medium"},
    )
    print("waiting...")
    return response.output_text
