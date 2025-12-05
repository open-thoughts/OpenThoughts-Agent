#!/usr/bin/env python3
"""
Train FastText model on datasets with positive and negative labels.
Simplified version of the train_fasttext_operator for standalone use.
"""

import logging
import os
import tempfile
from pathlib import Path
from typing import List, Optional

import fasttext
from datasets import Dataset, load_dataset, concatenate_datasets
from huggingface_hub import HfApi, hf_hub_download
from data.gcs_cache import gcs_cache

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def merge_datasets(dataset_ids: List[str], split: str = "train") -> Dataset:
    """
    Merge multiple HuggingFace datasets into one.

    Args:
        dataset_ids: List of HuggingFace dataset IDs
        split: Dataset split to load (default: "train")

    Returns:
        Combined dataset
    """
    datasets = []
    for dataset_id in dataset_ids:
        logger.info(f"Loading dataset: {dataset_id}")
        ds = load_dataset(dataset_id, split=split)
        datasets.append(ds)

    return concatenate_datasets(datasets)

@gcs_cache()
def train_fasttext_model(
    positive_dataset: Dataset,
    negative_dataset: Dataset,
    text_column: str,
    save_path: str,
    hf_repo_id: Optional[str] = None,
    dim: int = 256,
    epoch: int = 3,
    lr: float = 0.1,
    word_ngrams: int = 3,
    min_count: int = 3,
    tmpdir: Optional[str] = None,
) -> str:
    """
    Train a FastText supervised model on positive and negative examples.

    Args:
        positive_dataset: Dataset with positive examples
        negative_dataset: Dataset with negative examples
        text_column: Column name containing text data
        save_path: Path to save the trained model
        hf_repo_id: Optional HuggingFace repo ID to upload model
        dim: Dimension of word vectors (default: 256)
        epoch: Number of training epochs (default: 3)
        lr: Learning rate (default: 0.1)
        word_ngrams: Word N-grams parameter (default: 3)
        min_count: Minimum count of words (default: 3)
        tmpdir: Temporary directory for training files

    Returns:
        Path where the model was saved
    """
    with tempfile.TemporaryDirectory(dir=tmpdir) as temp_dir:
        train_file = os.path.join(temp_dir, "train.txt")
        model_file = os.path.join(temp_dir, "model.bin")

        # Write training data in FastText format
        logger.info(f"Writing training data to {train_file}")
        with open(train_file, "w", encoding="utf-8") as f:
            # Write positive examples
            for example in positive_dataset:
                text = example[text_column].replace("\n", " ")
                f.write(f"__label__positive {text}\n")

            # Write negative examples
            for example in negative_dataset:
                text = example[text_column].replace("\n", " ")
                f.write(f"__label__negative {text}\n")

        logger.info(f"Training FastText model with {len(positive_dataset)} positive and {len(negative_dataset)} negative examples")

        # Train the model
        model = fasttext.train_supervised(
            input=train_file,
            dim=dim,
            epoch=epoch,
            lr=lr,
            wordNgrams=word_ngrams,
            minCount=min_count,
        )

        # Save to temporary file
        model.save_model(model_file)
        logger.info(f"Model trained and saved to temporary file: {model_file}")

        # Copy to final destination
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        with open(model_file, "rb") as src, open(save_path, "wb") as dst:
            dst.write(src.read())

        logger.info(f"Model saved to: {save_path}")

        # Upload to HuggingFace if specified
        if hf_repo_id:
            try:
                logger.info(f"Uploading model to HuggingFace: {hf_repo_id}")
                api = HfApi()
                api.create_repo(
                    repo_id=hf_repo_id,
                    private=True,
                    exist_ok=True,
                )
                api.upload_file(
                    path_or_fileobj=model_file,
                    path_in_repo="model.bin",
                    repo_id=hf_repo_id,
                )
                logger.info(f"Model uploaded to HuggingFace: {hf_repo_id}")
            except Exception as e:
                logger.error(f"Failed to upload model to HuggingFace: {str(e)}")

    return save_path


def get_fasttext_model(
    hf_repo_id: str,
    positive_dataset: Optional[Dataset] = None,
    negative_dataset: Optional[Dataset] = None,
    text_column: str = "text",
    dim: int = 256,
    epoch: int = 3,
    lr: float = 0.1,
    word_ngrams: int = 3,
    min_count: int = 3,
    tmpdir: Optional[str] = None,
) -> fasttext.FastText._FastText:
    """
    Get FastText model - download from HuggingFace if exists, otherwise train and upload.

    Args:
        hf_repo_id: HuggingFace repo ID to check/upload model
        positive_dataset: Dataset with positive examples (required if training)
        negative_dataset: Dataset with negative examples (required if training)
        text_column: Column name containing text data (default: "text")
        dim: Dimension of word vectors (default: 256)
        epoch: Number of training epochs (default: 3)
        lr: Learning rate (default: 0.1)
        word_ngrams: Word N-grams parameter (default: 3)
        min_count: Minimum count of words (default: 3)
        tmpdir: Temporary directory for training files

    Returns:
        Loaded FastText model
    """
    # Try to download from HuggingFace
    try:
        logger.info(f"Checking if model exists in HuggingFace: {hf_repo_id}")
        model_path = hf_hub_download(
            repo_id=hf_repo_id,
            filename="model.bin",
            cache_dir=tmpdir,
        )
        logger.info(f"Model found! Downloaded from HuggingFace: {model_path}")
        return fasttext.load_model(model_path)

    except Exception as e:
        logger.info(f"Model not found in HuggingFace: {str(e)}")
        logger.info("Training new model...")

        # Validate training requirements
        if positive_dataset is None or negative_dataset is None:
            raise ValueError(
                "Model not found in HuggingFace and training datasets not provided. "
                "Please provide positive_dataset and negative_dataset to train a new model."
            )

        # Use temporary file for training
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False, dir=tmpdir) as tmp_file:
            save_path = tmp_file.name

        # Train the model
        model_path = train_fasttext_model(
            positive_dataset=positive_dataset,
            negative_dataset=negative_dataset,
            text_column=text_column,
            save_path=save_path,
            hf_repo_id=hf_repo_id,
            dim=dim,
            epoch=epoch,
            lr=lr,
            word_ngrams=word_ngrams,
            min_count=min_count,
            tmpdir=tmpdir,
        )

        logger.info(f"Model trained and available at: {model_path}")
        return fasttext.load_model(model_path)


def main():
    """
    Train and return a FastText model using positive and negative samples.
    """
    import sys
    sys.path.append(str(Path(__file__).parent.parent.parent))
    from data.freelancer_filtering.embedding_filter_mean import create_positive_negative_samples
    from data.commons import get_all_dcagent_positive_samples, get_code_contests_negative_samples
    
    # Create positive and negative samples
    logger.info("Creating positive and negative samples...")
    positive_samples, negative_samples = create_positive_negative_samples()
    
    # Convert to Dataset format
    from datasets import Dataset
    positive_dataset = Dataset.from_dict({"text": positive_samples})
    negative_dataset = Dataset.from_dict({"text": negative_samples})
    
    logger.info(f"Created {len(positive_dataset)} positive and {len(negative_dataset)} negative samples")
    
    # Train the model
    model_path = train_fasttext_model(
        positive_dataset=positive_dataset,
        negative_dataset=negative_dataset,
        text_column="text",
        save_path="./models/fasttext_model.bin",
        hf_repo_id=None,  # Set to None to skip HuggingFace upload
        dim=256,
        epoch=5,  # Increased epochs for better training
        lr=0.1,
        word_ngrams=3,
        min_count=2,  # Lower min_count for more vocabulary
    )
    
    logger.info(f"Model trained and saved to: {model_path}")
    
    # Load and return the trained model
    model = fasttext.load_model(model_path)
    return model


if __name__ == "__main__":
    main()
