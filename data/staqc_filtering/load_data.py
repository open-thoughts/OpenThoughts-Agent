#!/usr/bin/env python3
"""
Shared data loading for Stack Overflow filtering
"""

from typing import List

from data.stackexchange.generate_codereview import download_and_extract_dataset, parse_posts_xml, extract_questions_from_data
from data.gcs_cache import gcs_cache


@gcs_cache()
def load_staqc_questions() -> List[str]:
    """Load Stack Overflow questions"""
    posts_xml_path = download_and_extract_dataset("https://archive.org/download/stackexchange/stackoverflow.com-Posts.7z")
    questions_data = parse_posts_xml(posts_xml_path, limit=200_000)
    questions = extract_questions_from_data(questions_data)
    questions = list(set(questions))  # dedupe
    print(f"Loaded {len(questions)} unique Stack Overflow questions")
    return questions
