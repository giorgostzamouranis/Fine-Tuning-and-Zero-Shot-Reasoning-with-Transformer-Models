# Fine-Tuning and Zero-Shot Reasoning with Transformer Models

This repository explores the practical application of transformer-based language models using Hugging Face Transformers and Datasets libraries. It covers both **fine-tuning** and **zero-shot inference** across several NLP tasks involving sentiment classification, commonsense reasoning, semantic similarity, and truthfulness evaluation.

---

##  Table of Contents

- [Part A: Fine-Tuning on Yelp Dataset](#part-a-fine-tuning-on-yelp-dataset)
- [Part B: Zero-Shot Transfer and Reasoning](#part-b-zero-shot-transfer-and-reasoning)
  - [B1: PIQA - Commonsense Reasoning](#b1-piqa---commonsense-reasoning)
  - [B2: TruthfulQA - Answer Validity & Semantic Similarity](#b2-truthfulqa---answer-validity--semantic-similarity)
  - [B3: Winogrande - Fill-in-the-Blank Commonsense](#b3-winogrande---fill-in-the-blank-commonsense)
- [ Model Comparison & Conclusions](#model-comparison--conclusions)
- [ Dependencies](#dependencies)

---

## Part A: Fine-Tuning on Yelp Dataset

A pre-trained `distilbert-base-uncased` model was fine-tuned on a balanced subset (300 samples) of the [Yelp Polarity Dataset](https://huggingface.co/datasets/yelp_polarity). Multiple training experiments were performed by varying:

- Epochs: 5, 10, 15
- Learning rates: `1e-5`, `3e-5`, `5e-5`
- Batch sizes: 8, 16, 32

###   Key Findings

| Epochs | LR    | Batch Size | Accuracy | Notes                             |
|--------|-------|------------|----------|-----------------------------------|
| 5      | 3e-5  | 16         | 91.3%    | Undertrained                      |
| 10     | 3e-5  | 16         | 92.3%    | Stable improvement                |
| 15     | 3e-5  | 16         | 92.7–93% | Best result, no overfitting       |
| 15     | 1e-5  | 16         | 92.3%    | Slower convergence                |
| 15     | 5e-5  | 16         | 93%      | Risk of overfitting               |
| 15     | 3e-5  | 8          | 91.3%    | High variance, less stable        |
| 15     | 3e-5  | 32         | 92.0%    | Smooth but slightly lower result  |

**Conclusion:** A learning rate of `3e-5` and batch size of `16` yielded the best trade-off between speed and generalization.

---

## Part B: Zero-Shot Transfer and Reasoning

### B1: PIQA - Commonsense Reasoning

Evaluated 5 zero-shot models using the [PIQA dataset](https://huggingface.co/datasets/piqa). Flan-T5 significantly outperformed others.

| Model                          | Accuracy |
|--------------------------------|----------|
| `roberta-large-mnli`          | 54%      |
| `facebook/bart-large-mnli`    | 47%      |
| `google/flan-t5-large`        | 80%      |
| `allenai/unifiedqa-t5-large`  | 43%      |
| `nli-deberta-v3-large`        | 67%      |

---

### B2: TruthfulQA - Answer Validity & Semantic Similarity

Paired 5 QA models with 6 semantic similarity models (Sentence Transformers). The best answer was validated using cosine similarity ≥ `0.70`.

**Best Combination:** `Flan-T5` with `multi-qa-MiniLM-L6-cos-v1` → Accuracy: **32.2%**

---

### B3: Winogrande - Fill-in-the-Blank Commonsense

Evaluated model ability to resolve ambiguous fill-in-the-blank scenarios from the [Winogrande dataset](https://huggingface.co/datasets/winogrande).

| Model                             | Accuracy |
|----------------------------------|----------|
| `DeepPavlov/roberta-winogrande`  | 55%      |
| `facebook/bart-large-mnli`       | 77%      |
| `roberta-large-mnli`             | 62%      |

**Conclusion:** Despite being fine-tuned, DeepPavlov underperformed compared to zero-shot inference with BART.

---

## Model Comparison & Conclusions

- **Fine-tuned transformers** (like DistilBERT) work well for targeted classification tasks with sufficient supervision.
- **Instruction-tuned models** (like Flan-T5) generalize better in zero-shot scenarios requiring reasoning.
- **Semantic similarity models** like `multi-qa-MiniLM` enhance evaluation quality when paired with QA models.

---

##   Dependencies

Install the required libraries:

```bash
pip install transformers datasets evaluate sentence-transformers
```
---
##  Folder Structure
```bash
├── colab_notebook.ipynb      # Main notebook with all code and experiments
├── README.md                 # Project documentation
├── results/                  # Training logs and accuracy comparisons
└── models/                   # Saved fine-tuned models (optional)
```
---
## Author
Created as part of Neural Networks and Deep Learning lab assignment focused on applying and analyzing transformer-based models in practical tasks.

