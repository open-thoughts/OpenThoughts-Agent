#!/usr/bin/env python3
"""
Generate Stack Overflow Code Review dataset in sandboxes style
"""

import os
import re
import json
import xml.etree.ElementTree as ET
import tempfile
import shutil
import urllib.request
import sys
from pathlib import Path
from typing import List, Dict, Any
import py7zr

# Import from parent package
from data.commons import upload_tasks_to_hf, generate_tasks_from_questions, subsample_tasks_directory
from scripts.harbor.run_and_export_traces import run_dataset_to_traces
from data.commons import upload_traces_to_hf
from data.gcs_cache import gcs_cache

@gcs_cache()
def download_and_extract_dataset(url: str) -> str:
    """Download and extract the Stack Exchange Code Review dataset"""
    print("Downloading Stack Exchange Code Review dataset...")

    # Create temporary directory for download
    temp_dir = Path(tempfile.mkdtemp(prefix="stackexchange_"))
    archive_path = temp_dir / "codereview.stackexchange.com.7z"

    # Download the 7z archive
    urllib.request.urlretrieve(url, archive_path)

    print(f"Downloaded to: {archive_path}")

    # Extract the 7z archive
    extract_dir = temp_dir / "extracted"
    extract_dir.mkdir()

    with py7zr.SevenZipFile(archive_path, mode='r') as z:
        z.extractall(path=extract_dir)

    print(f"Extracted to: {extract_dir}")

    # Find the Posts.xml file
    posts_xml = None
    for root, dirs, files in os.walk(extract_dir):
        for file in files:
            if file == "Posts.xml":
                posts_xml = Path(root) / file
                break
        if posts_xml:
            break

    if not posts_xml:
        raise FileNotFoundError("Could not find Posts.xml in extracted archive")

    print(f"Found Posts.xml at: {posts_xml}")
    return posts_xml

@gcs_cache()
def parse_posts_xml(posts_xml_path: str, limit: int = None) -> List[Dict[str, Any]]:
    """Parse Posts.xml and extract questions"""
    print("Parsing Posts.xml for questions...")

    questions = []

    # Parse XML incrementally to handle large files
    context = ET.iterparse(posts_xml_path, events=('start', 'end'))
    context = iter(context)
    event, root = next(context)

    count = 0
    for event, elem in context:
        if event == 'end' and elem.tag == 'row':
            # Check if this is a question (PostTypeId=1)
            post_type = elem.get('PostTypeId')
            if post_type == '1':  # Question
                title = elem.get('Title', '')
                body = elem.get('Body', '')
                tags = elem.get('Tags', '')
                score = int(elem.get('Score', 0))

                # Filter for reasonable quality questions
                if (title and body and
                    len(body) > 100 and  # Minimum body length
                    score >= 0 and      # Non-negative score
                    '<code>' in body):   # Contains code

                    # Clean HTML tags from body
                    import html
                    body_text = html.unescape(body)
                    body_text = re.sub(r'<[^>]+>', ' ', body_text)
                    body_text = re.sub(r'\s+', ' ', body_text).strip()

                    questions.append({
                        'id': elem.get('Id'),
                        'title': html.unescape(title),
                        'body': body_text,
                        'tags': tags,
                        'score': score
                    })

                    count += 1
                    if count % 1000 == 0:
                        print(f"Processed {count} questions...")

            # Clear the element to save memory
            elem.clear()
            
            if limit and count >= limit:
                break

    print(f"Extracted {len(questions)} questions")
    return questions

@gcs_cache()
def extract_questions_from_data(questions_data: List[Dict[str, Any]]) -> List[str]:
    """Extract question strings from the parsed data"""
    questions = []
    for data in questions_data:
        title = data["title"]
        body = data["body"]
        tags = data["tags"]

        # Combine title, body and tags into a single question string
        question = f"""
{body}
"""
        questions.append(question)
    return questions


def main() -> None:
    """Main function - coordinates the pipeline with temp directories"""
    # Load data
    posts_xml_path = download_and_extract_dataset("https://archive.org/download/stackexchange/codereview.stackexchange.com.7z")
    questions_data = parse_posts_xml(posts_xml_path)
    questions = extract_questions_from_data(questions_data)
    final_dataset_dir = generate_tasks_from_questions(questions, "codereview")
    subsampled_dataset_dir = subsample_tasks_directory(final_dataset_dir, 10_000)
    upload_tasks_to_hf(subsampled_dataset_dir, "mlfoundations-dev/stackexchange-codereview-sandboxes")
    hf_dataset = run_dataset_to_traces(subsampled_dataset_dir, model_name="gpt-5-nano-2025-08-07", agent_name="terminus-2", n_concurrent=1024, agent_kwargs={"max_episodes": 8})
    upload_traces_to_hf(hf_dataset, "mlfoundations-dev/stackexchange-codereview-sandboxes-traces-terminus-2", "SFT")
if __name__ == "__main__":
    main()