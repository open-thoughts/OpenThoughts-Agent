import os
import re
import random
from typing import Dict
from bespokelabs import curator
from datasets import Dataset, load_dataset
from pydantic import BaseModel, Field
from typing import Literal

from datasets import load_dataset, concatenate_datasets

from data.commons import (
    upload_traces_to_hf
)


class Solve(BaseModel):
  output: str = Field(
    description="Solution to the original question")

class SolveGenerator(curator.LLM):

  def prompt(self, row: Dict):
    return f"Think step-by-step and solve the following question: {row['conversations'][0]['content']}"

  def parse(self, row: Dict, response: Solve):
    output = response.output
    conversations = [{'content': row['conversations'][0]['content'], 'role': 'user'}, {'content': output, 'role': 'assistant'}]
    return [{"conversations": conversations}]

if __name__ == "__main__":
    
    staqc_ot3_traces = load_dataset("DCAgent/staqc-ot3-100k-science-subset-traces-terminus-2", split="train")
    length_staqc_ot3_traces = len(staqc_ot3_traces)
    print(f"Length of staqc_ot3_traces: {length_staqc_ot3_traces}")
    staqc_traces = load_dataset("mlfoundations-dev/staqc-sandboxes-traces-terminus-2", split="train")
    length_staqc_traces = len(staqc_traces)
    print(f"Length of staqc_traces: {length_staqc_traces}")

    # compute how many examples to select from the end
    n_to_select = length_staqc_ot3_traces - length_staqc_traces

    # select the last n examples
    staqc_ot3_tail = staqc_ot3_traces.select(range(length_staqc_ot3_traces - n_to_select, length_staqc_ot3_traces))
    # staqc_ot3_tail = staqc_ot3_tail.select(range(10))
    print(f"Length of staqc_ot3_tail: {len(staqc_ot3_tail)}")

    analyzer = SolveGenerator(
      model_name="gpt-5-nano", response_format=Solve, batch=False)

    ot3_science_gpt_traces = analyzer(staqc_ot3_tail)
    upload_traces_to_hf(ot3_science_gpt_traces, "DCAgent/ot3-100k-science-subset-gpt-5-nano-traces", "SFT")

    staqc_ot3_traces_head = staqc_ot3_traces.select(range(length_staqc_traces))

    staqc_ot3_merged = concatenate_datasets([staqc_ot3_traces_head, ot3_science_gpt_traces])
    upload_traces_to_hf(staqc_ot3_merged, "DCAgent/staqc-ot3-100k-science-subset-gpt-5-nano-traces", "SFT")