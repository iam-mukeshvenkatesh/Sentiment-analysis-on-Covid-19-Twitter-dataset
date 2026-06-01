# Sentiment-analysis-on-Covid-19-Twitter-dataset

## Overview

This project performs three-class sentiment analysis (Positive / Neutral / Negative) on COVID-19 tweets using an ensemble of three models. The final prediction is made by averaging probability distributions across all three, achieving **94.95% accuracy** on the test set.

---

## Models

| Model | Architecture | Embeddings |
|---|---|---|
| Model 1 | CNN + BiLSTM | BERT (`bert-base-uncased`) |
| Model 2 | CNN + BiLSTM | GloVe (100d) |
| Model 3 | SVM (linear kernel) | Count Vectorizer (GPU via cuML) |

The ensemble averages the softmax probability outputs of all three models before making a final prediction.

---

## Dataset

**COVID-19 Twitter Dataset** by Arunavak Chakraborty — available on [Kaggle](https://www.kaggle.com/datasets/arunavakrchakraborty/covid19-twitter-dataset).

Covers tweets from April–September 2020 and April–June 2021. Labels (`pos`, `neu`, `neg`) are derived from VADER compound scores.

---

## Repository Structure

```
├── ensemble_edition_16_apr.ipynb   # Training notebook (all 3 models + ensemble)
├── GUI_Interface.py                # Streamlit web app for inference
├── requirements.txt                # Python dependencies (create this — see below)
└── README.md
```

> **Note:** Trained model files (`bert_cnn_bilstm.pt`, `glove_cnn_bilstm.pt`, `svm_model.pkl`, `vectorizer.pkl`) are not included due to size. See the notebook to retrain.

---

## Setup & Installation

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
pip install -r requirements.txt
```

**Key dependencies:**
- Python 3.9+
- PyTorch
- Transformers (HuggingFace)
- NLTK
- Streamlit
- scikit-learn
- cuML / cuDF (NVIDIA RAPIDS — required for GPU-accelerated SVM training)
- kagglehub

---

## Running the Web App

```bash
streamlit run GUI_Interface.py
```

Open `http://localhost:8501` in your browser. The app accepts any text input and returns a sentiment prediction with confidence scores for all three classes.

> The current GUI uses simulated predictions as a demo. To enable real inference, load the saved model files in the `load_models()` function inside `GUI_Interface.py`.

---

## Results

| Model | Accuracy |
|---|---|
| BERT-CNN-BiLSTM | ~93% |
| GloVe-CNN-BiLSTM | ~91% |
| SVM (GPU, linear) | **98.26%** |
| **Ensemble** | **94.95%** |

---

## Acknowledgements

- Dataset: [Arunavak Chakraborty on Kaggle](https://www.kaggle.com/datasets/arunavakrchakraborty/covid19-twitter-dataset)
- BERT: [HuggingFace Transformers](https://huggingface.co/bert-base-uncased)
- GPU acceleration: [NVIDIA RAPIDS (cuML)](https://rapids.ai/)

---
