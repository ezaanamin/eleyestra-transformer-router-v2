# Elyestra Router Dataset Builder & Transformer

A robust, end-to-end Machine Learning pipeline designed to build, clean, and train an intent-routing classifier for a personal AI assistant (Elyestra). This repository demonstrates the entire lifecycle of an ML project, from synthetic data augmentation and LLM-assisted dataset labeling to training a highly accurate Transformer model (DistilBERT) for intent classification.

## 🚀 Project Overview

The core goal of this project is to construct a production-ready routing classifier that directs a user's natural language queries into one of four domain-specific contexts:
- `GENERAL`: General world knowledge, abstract questions, or conversational filler.
- `PERSONAL_CONTEXT`: Questions requiring the user's personal history, goals, or opinions.
- `CODING_CONTEXT`: Technical tasks, code snippets, logs, and development requests.
- `AGENT`: Action-oriented tasks (e.g., sending emails, checking finances, managing calendar events, booking reminders).

Instead of relying solely on expensive, high-latency LLM calls for every message, this system trains a lightweight DistilBERT model to perform the routing in milliseconds, acting as a highly efficient traffic controller for the Elyestra AI ecosystem.

## 🧠 ML Pipeline & Architecture

This repository highlights a comprehensive ML engineering workflow:

1. **Dataset Construction & Augmentation**  
   - Generated thousands of diverse, synthetic training examples for specialized domains (calendar management, email operations, financial tracking).
   - Used local LLMs (Qwen via Ollama) to augment data by anchoring synthetic prompts to realistic contexts while strictly preserving the user's unique typing style (typos, casing, tone).

2. **LLM-Assisted Labeling**  
   - Built a robust, parallelized labeling pipeline (`label_dataset.py`) to classify thousands of unlabeled queries.
   - Designed a highly specific prompt-engineering schema with strict output parsing to assign probabilities and fallback handling for ambiguous inputs.

3. **Data Cleaning & Class Balancing**  
   - Handled duplicated records, parsed raw JSON exports from Google Calendar, and aggregated real-world financial transaction data into ML-ready formats.
   - Formatted inputs specifically to prevent data leakage and balance the distribution across the four intent categories.

4. **Transformer Training & Validation**  
   - Engineered a custom training loop in a Jupyter Notebook using Hugging Face's `transformers` and `datasets` libraries.
   - Fine-tuned a `distilbert-base-uncased` model to achieve high multi-class precision, recall, and F1 scores.
   - Evaluated training curves, confusion matrices, and validation loss to prevent overfitting on the synthetic distribution.

5. **Real-World Testing & Evaluation**  
   - Performed manual inference testing against unseen, real-world conversational fragments to ensure the model generalizes beyond its training set.

## 📂 Repository Structure

*Note: All private datasets, logs, model checkpoints, and personal outputs have been excluded from this public repository via `.gitignore` to protect sensitive information while preserving the source code.*

- `src/label_dataset.py`: Core labeling script utilizing local Ollama instances for deterministic, schema-enforced dataset annotation.
- `src/generate_*_prompts.py`: Domain-specific scripts (Finance, Email, Calendar/Agent) used for targeted dataset augmentation and synthetic prompt generation.
- `src/format_calendar_data.py`: Preprocessing script that normalizes raw Google Calendar `.json`/`.ics` exports into clean, structured data for ML consumption.
- `notebooks/Elyestra_router.ipynb`: The primary ML notebook documenting the dataset loading, tokenization, DistilBERT training phase, evaluation metrics, and manual inference testing.
- `.env.example`: Configuration template for API keys and local LLM ports.

## 🛠️ Tech Stack
- **Python 3.10+**
- **Hugging Face Ecosystem:** `transformers`, `datasets` (DistilBERT)
- **Data Processing:** `pandas`, `numpy`
- **LLM Integration:** Ollama (Qwen 3), `requests`

## 🔒 Privacy & Security Note
This repository contains the full source code and engineering logic for the pipeline. However, to maintain privacy, all real datasets (CSV/JSON/JSONL), private Google Calendar exports, raw financial transactions, logs, and sensitive Jupyter Notebook outputs have been explicitly omitted.
