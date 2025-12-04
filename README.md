# BESSTIE Project – DeepNLP 2025/2026

Project for the Polito "Deep Natural Language Processing" course.

**Topic:** BESSTIE – Sentiment and Sarcasm Classification for Varieties of English  
Paper: “BESSTIE: A Benchmark for Sentiment and Sarcasm Classification for Varieties of English”

## Repo structure

- `src/`
  - `data_utils.py` – load the BESSTIE dataset from Hugging Face
  - `models.py` – utilities to load HF models + tokenizers
  - `train_eval.py` – generic training loop using HuggingFace Trainer
- `notebooks/`
  - will contain the Colab notebooks (exploration, baselines, extensions)
- `requirements.txt` – main Python dependencies

## Goal

1. Reproduce part of the BESSTIE experiments (sentiment + sarcasm).
2. Analyse performance across English varieties (en-AU, en-IN, en-UK).
3. Add at least one extension (e.g., cross-variety or cross-domain experiments).

