import os 
from datasets import load_dataset, concatenate_datasets
from data.commons import  upload_traces_to_hf

# Convert ot3_traces to STAQC format
def convert_conversation(example):
    new_conversations = []
    for turn in example["conversations"]:
        role = turn.get("from", "")
        content = turn.get("value", "")
        role = "user" if role == "human" else "assistant"
        new_conversations.append({
            "role": role,
            "content": content
        })
    example["conversations"] = new_conversations
    return example


if __name__ == "__main__":

    staqc_traces = load_dataset("mlfoundations-dev/staqc-sandboxes-traces-terminus-2", split="train")
    # ot3_traces = load_dataset("open-thoughts/OpenThoughts3-1.2M", split="train")
    ot3_traces = load_dataset("/data1/hbansal/OpenThoughts3-1.2M/", split="train")

    ot3_traces = ot3_traces.shuffle(seed=42).select(range(100_000))
    
    ## filter examples with domain == science
    # ot3_traces = ot3_traces.filter(lambda x: x["domain"] == "science")

    ot3_traces = ot3_traces.map(convert_conversation, desc="Converting OT3 format")

    ot3_traces = ot3_traces.remove_columns(
        [col for col in ot3_traces.column_names if col != "conversations"]
    )

    staqc_traces = staqc_traces.remove_columns(
        [col for col in staqc_traces.column_names if col != "conversations"]
    )

    ## convert the conversations column to the expected type
    staqc_ot3_merged = concatenate_datasets([staqc_traces, ot3_traces])

    ## upload the dataset to hf
    upload_traces_to_hf(staqc_ot3_merged, "DCAgent/staqc-ot3-100k-traces-terminus-2", "SFT")