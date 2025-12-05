from bespokelabs import curator
from typing import List
import re
from data.gcs_cache import gcs_cache

class DockerGenerator(curator.LLM):
  def prompt(self, prompt: str) -> str:
    return f"""
    Generate a Dockerfile for the following prompt:
    {prompt}
    Give me only the Dockerfile, do not include any other text. Put the Dockerfile in 
    a code block with the language set to Dockerfile. Also install tmux in the Dockerfile.

    EXAMPLE:
    ```dockerfile
    FROM ubuntu:24.04
    WORKDIR /app
    RUN apt-get update -y && \
    apt-get install -y tmux
    RUN apt-get update && apt-get install -y python3 python3-pip && rm -rf /var/lib/apt/lists/*
    ```
    
    """

  def parse(self, prompt: str, response: str):
    # Extract content between ```dockerfile ... ```
    match = re.search(r"```dockerfile\s*(.*?)```", response, re.DOTALL | re.IGNORECASE)
    dockerfile = match.group(1).strip() if match else response.strip()
    return {"prompt": prompt, "dockerfile": dockerfile}

@gcs_cache()
def gpt_convert_prompt_to_dockerfile_batched(prompts: List[str]) -> List[str]:
    """
    Convert a prompt to a Dockerfile.
    """
    generator = DockerGenerator(
        model_name="gpt-5-nano-2025-08-07"
    )
    dataset = generator(prompts).dataset
    assert prompts == dataset["prompt"]
    dockerfiles = dataset["dockerfile"]
    return dockerfiles