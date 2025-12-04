# src/data_utils.py

from datasets import load_dataset

def load_besstie():
    """
    Load the BESSTIE dataset from Hugging Face.

    Returns:
        dataset: a datasets.DatasetDict with splits and metadata.
    """
    dataset = load_dataset("unswnlporg/BESSTIE")
    return dataset
