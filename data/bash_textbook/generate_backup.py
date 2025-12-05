#!/usr/bin/env python3
"""
PDF to Questions Generator - Based on organic chemistry PDF processing pipeline
Implements atomic functions following the pattern from YAML config and completions operator
"""

import copy
import io
import logging
import os
import re
import tempfile
from io import BytesIO
from typing import Dict, List, Optional
from pathlib import Path

import fitz
import requests
from datasets import Dataset, Features, Sequence, Value
from tqdm import tqdm

from data.commons import generate_tasks_from_questions, create_task_from_dockerfiles_and_questions, subsample_tasks_directory, upload_tasks_to_hf, upload_traces_to_hf
from data.completions import generate_questions_from_text, improve_questions, validate_questions, classify_domain
from data.gcs_cache import gcs_cache
from data.commons import upload_tasks_to_hf, generate_tasks_from_questions, subsample_tasks_directory
# from scripts.harbor.run_and_export_traces import run_dataset_to_traces
# from scripts.harbor.run_and_export_traces import run_dataset_to_traces  # Skip for initial test


def create_url_dataset(urls: List[str]) -> Dataset:
    """
    Create a dataset from a list of URLs.

    Args:
        urls (List[str]): List of PDF URLs

    Returns:
        Dataset: HuggingFace dataset with columns:
            - url: source URL
    """
    return Dataset.from_dict({"url": urls})


@gcs_cache()
def download_pdfs(dataset: Dataset) -> Dataset:
    """
    Download PDFs from URLs and add serialized content to dataset.
    Uses parallel processing via dataset.map()

    Args:
        dataset (Dataset): Dataset containing URLs in 'url' column

    Returns:
        Dataset: Original dataset with additional columns:
            - pdf_bytes: serialized PDF content
            - filename: extracted filename from URL
            - success: whether download succeeded
    """

    def download_single_pdf(example):
        url = example["url"]
        # Extract filename from URL
        filename = url.split("/")[-1]
        if not filename.lower().endswith(".pdf"):
            filename += ".pdf"

        # Download PDF with progress tracking disabled for parallel processing
        headers = {}  # Add any required headers here
        try:
            response = requests.get(url, headers=headers, stream=True, timeout=30)
            
            if response.status_code == 200:
                # Get the PDF content as bytes
                pdf_bytes = response.content

                return {
                    **example,
                    "pdf_bytes": pdf_bytes,
                    "filename": filename,
                    "success": True,
                }
        except Exception as e:
            print(f"Error downloading {url}: {str(e)}")
            
        return {**example, "pdf_bytes": b"", "filename": filename, "success": False}

    # Use map to process URLs in parallel
    dataset = dataset.map(download_single_pdf, num_proc=8, desc="Downloading PDFs")
    return dataset


def get_pdf_page_count(url_or_bytes):
    """Get page count from PDF URL or bytes"""
    try:
        if isinstance(url_or_bytes, str):
            response = requests.get(url_or_bytes, stream=True, timeout=30)
            pdf_data = BytesIO(response.content)
        else:
            pdf_data = BytesIO(url_or_bytes)
            
        pdf_document = fitz.open(stream=pdf_data, filetype="pdf")
        page_count = pdf_document.page_count
        pdf_document.close()
        
        # TODO - nodes will OOM if the text is too long
        return min(page_count, 500)
    except Exception as e:
        print(f"Error getting page count: {str(e)}")
        return 0


@gcs_cache()
def get_all_page_counts(dataset: Dataset) -> Dataset:
    """Add page count to each PDF in the dataset"""
    def f(x):
        try:
            if x["success"] and x["pdf_bytes"]:
                page_count = get_pdf_page_count(x["pdf_bytes"])
                x["page_count"] = page_count
            else:
                x["page_count"] = 0
        except Exception as e:
            print(f"Error getting page count for {x['url']}: {str(e)}")
            x["page_count"] = 0
        return x

    dataset = dataset.map(f, desc="Getting page counts")
    return dataset


@gcs_cache()
def expand_and_extract_pages(dataset: Dataset, page_count_column: str = "page_count") -> Dataset:
    """
    Expands a dataset by creating new rows for each page number and extracts
    individual pages from PDFs. Expects dataset to have 'pdf_bytes' column
    and the specified page count column.
    """
    expanded_data = []

    # Process each PDF document
    for example in tqdm(dataset, desc="Extracting pages"):
        if not example.get("success", True) or not example["pdf_bytes"]:
            continue

        try:
            # Open PDF from bytes once per document
            pdf_stream = BytesIO(example["pdf_bytes"])
            src_pdf = fitz.open(stream=pdf_stream, filetype="pdf")
            page_count = int(example[page_count_column])

            # Create a new row for each page
            for page_num in range(1, min(page_count + 1)):  # Limit to 50 pages
                example_copy = copy.deepcopy(example)
                example_copy["pdf_bytes"] = "REMOVED"  # Remove to save memory
                example_copy["page_number"] = page_num
                
                # Create new PDF with single page
                dst_pdf = fitz.open()
                dst_pdf.insert_pdf(src_pdf, from_page=page_num - 1, to_page=page_num - 1)

                # Save to bytes
                output_stream = BytesIO()
                dst_pdf.save(output_stream)
                example_copy["page_bytes"] = output_stream.getvalue()

                # Clean up
                dst_pdf.close()
                expanded_data.append(example_copy)

            # Clean up the source PDF
            src_pdf.close()

        except Exception as e:
            print(f"Error processing PDF from {example.get('url', 'unknown URL')}: {str(e)}")
            continue

    # Create new dataset from expanded data
    return Dataset.from_list(expanded_data)


@gcs_cache()
def extract_text_from_page_bytes(dataset: Dataset) -> Dataset:
    """Extract text from PDF page bytes using OCR/text extraction"""
    def extract_text(example):
        try:
            if example.get("page_bytes"):
                pdf_stream = BytesIO(example["page_bytes"])
                pdf = fitz.open(stream=pdf_stream, filetype="pdf")
                page = pdf[0]  # Single page
                text = page.get_text()
                pdf.close()
                
                example["output_extraction"] = text
            else:
                example["output_extraction"] = ""
        except Exception as e:
            print(f"Error extracting text: {str(e)}")
            example["output_extraction"] = ""
        return example
    
    return dataset.map(extract_text, desc="Extracting text from pages")




def process_extracted_questions(dataset: Dataset) -> Dataset:
    """
    Processes each example from the input dataset.
    Looks for multiple "QUESTION: " blocks and expands into separate rows.
    """
    new_rows = []

    # Iterate over each example in the dataset
    for ex in dataset:
        # Safely get the content from "question_choices_solutions"
        entry = ex.get("question_choices_solutions", "")
        entry = entry.strip() if isinstance(entry, str) else ""

        # If there's no text, create a single row with default values
        if not entry:
            row = dict(ex)  # copy original columns
            row["extracted_question"] = "NO QUESTION DETECTED"
            row["extracted_answer_choices"] = ["NO CHOICES AVAILABLE"]
            row["extracted_solution"] = "NO ANSWER DETECTED"
            new_rows.append(row)
            continue

        # Split the text by the question marker
        sections = entry.split("\nQUESTION: ")
        found_any_question = False

        for section in sections:
            section = section.strip()
            if not section:
                continue  # skip empty splits

            found_any_question = True

            # "parts[0]" is the question, "parts[1]" includes the answer choices + solution
            parts = section.split("\nANSWER CHOICES: ")
            question_text = parts[0].strip() or "NO QUESTION DETECTED"

            if len(parts) > 1:
                # We have some answer choices; split for solution
                answer_solution_parts = parts[1].split("\nSOLUTION: ")
                answer_choices_raw = answer_solution_parts[0].strip()
                solution = (
                    answer_solution_parts[1].strip()
                    if len(answer_solution_parts) > 1
                    else "NO ANSWER DETECTED"
                )
            else:
                # No explicit "ANSWER CHOICES:" block
                answer_choices_raw = "free response"
                solution = "NO ANSWER DETECTED"

            # Always ensure question is a string
            question_text = question_text if question_text else "NO QUESTION DETECTED"

            # Convert answer choices into a list of strings
            if "|||" in answer_choices_raw:
                # Split by delimiter
                answer_choices_list = [
                    choice.strip()
                    for choice in answer_choices_raw.split(" ||| ")
                    if choice.strip()
                ]
                # If the split is empty after stripping, use a default
                if not answer_choices_list:
                    answer_choices_list = ["NO CHOICES AVAILABLE"]
            elif (
                answer_choices_raw.lower() == "free response" or not answer_choices_raw
            ):
                answer_choices_list = ["NO CHOICES AVAILABLE"]
            else:
                answer_choices_list = [answer_choices_raw]

            # Ensure the solution is a non-empty string
            solution = solution if solution else "NO ANSWER DETECTED"

            # Build a new row preserving the original columns
            row = dict(ex)
            row["extracted_question"] = question_text.replace("QUESTION:", "").strip()
            row["extracted_answer_choices"] = answer_choices_list
            row["extracted_solution"] = solution.replace("SOLUTION:", "").strip()
            new_rows.append(row)

        # If we never found any "QUESTION: " in the text,
        # add a single default row
        if not found_any_question:
            row = dict(ex)
            row["extracted_question"] = "NO QUESTION DETECTED"
            row["extracted_answer_choices"] = ["NO CHOICES AVAILABLE"]
            row["extracted_solution"] = "NO ANSWER DETECTED"
            new_rows.append(row)

    # Convert the list of dictionaries into a new Hugging Face Dataset
    new_dataset = Dataset.from_list(new_rows)
    return new_dataset




def filter_valid_qas(dataset: Dataset) -> Dataset:
    """
    Filters out rows that don't meet quality conditions
    """
    def is_valid(example):
        return (
            example.get("extracted_solution", "") != "NO ANSWER DETECTED"
            and example.get("extracted_solution", "") not in ["free response answer", '"free response"', "free response"]
            and example.get("extracted_question", "") != "NO QUESTION DETECTED"
            and example.get("extracted_solution", "") != '"NO ANSWER DETECTED"'
        )

    filtered_dataset = dataset.filter(is_valid)
    return filtered_dataset




def filter_by_validation(dataset: Dataset) -> Dataset:
    """Filter dataset to keep only validated questions"""
    return dataset.filter(lambda x: x.get("qa_validation_outputs", False))


def convert_to_instructions(dataset: Dataset) -> List[str]:
    """Convert dataset to list of instruction strings"""
    instructions = []
    for example in dataset:
        if example.get("improved_question_solution"):
            instructions.append(example["improved_question_solution"])
    return instructions


def main() -> None:
    """Main pipeline function following atomic pattern"""
    
    # Bash textbook and documentation URLs
    bash_textbook_urls = [
        "https://www.tldp.org/LDP/Bash-Beginners-Guide/Bash-Beginners-Guide.pdf",
        "https://www.gnu.org/software/bash/manual/bash.pdf",
        "https://www.tldp.org/LDP/abs/abs-guide.pdf",  # Advanced Bash Scripting Guide
        "https://linuxcommand.org/tlcl.pdf",  # The Linux Command Line
        "https://www.freecodecamp.org/news/content/images/2022/03/Bash-Scripting-Tutorial.pdf",
        "https://www.cyberciti.biz/media/new/faq/2009/11/bash_reference_manual.pdf",
        # Add more bash textbook URLs as needed
    ]
    
    print("Creating URL dataset...")
    url_dataset = create_url_dataset(bash_textbook_urls)
    
    print("Downloading PDFs...")
    pdf_dataset = download_pdfs(url_dataset)
    
    print("Getting page counts...")
    page_count_dataset = get_all_page_counts(pdf_dataset)
    
    print("Extracting individual pages...")
    pages_dataset = expand_and_extract_pages(page_count_dataset)
    
    
    print("Extracting text from pages...")
    text_dataset = extract_text_from_page_bytes(pages_dataset)
    
    print("Generating questions from text...")
    breakpoint()
    questions = generate_questions_from_text(text_dataset)
    breakpoint()
    final_dataset_dir = generate_tasks_from_questions(questions, "bash_textbook")
    subsampled_dataset_dir = subsample_tasks_directory(final_dataset_dir, 10_000)
    upload_tasks_to_hf(subsampled_dataset_dir, "DCAgent/bash_textbook_tasks")
    breakpoint()
    # hf_dataset = run_dataset_to_traces(subsampled_dataset_dir, model_name="gpt-5-nano-2025-08-07", agent_name="terminus-2", n_concurrent=256, agent_kwargs={"max_episodes": 8})
    # upload_traces_to_hf(hf_dataset, "DCAgent/bash_textbook_tasks_traces", "SFT")

if __name__ == "__main__":
    main()