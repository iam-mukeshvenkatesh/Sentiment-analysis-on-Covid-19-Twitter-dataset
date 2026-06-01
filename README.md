# Sentiment-analysis-on-Covid-19-Twitter-dataset

## Overview

This project performs three-class sentiment analysis (Positive / Neutral / Negative) on COVID-19 tweets using an ensemble of three models. The final prediction is made by averaging probability distributions from all three models for robust sentiment classification.

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
├── ensemble_edition.ipynb      # Training notebook (all 3 models + ensemble)
├── GUI_Interface.py            # Streamlit web app for inference
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git ignore rules
└── README.md                   # This file
```

> **Note:** Trained model files (`bert_cnn_bilstm.pt`, `glove_cnn_bilstm.pt`, `svm_model.pkl`, `vectorizer.pkl`) are not included due to size. See the notebook to retrain.

---

## Setup & Installation

### 1. Clone the Repository

```bash
git clone https://github.com/iam-mukeshvenkatesh/Sentiment-analysis-on-Covid-19-Twitter-dataset.git
cd Sentiment-analysis-on-Covid-19-Twitter-dataset
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**Key dependencies:**
- Python 3.9+
- PyTorch (for deep learning models)
- Transformers (HuggingFace BERT)
- NLTK (text preprocessing)
- Streamlit (web interface)
- scikit-learn (machine learning utilities)
- cuML / cuDF (NVIDIA RAPIDS — GPU-accelerated SVM training, optional but recommended)
- kagglehub (dataset downloading)

---

## Getting Trained Models

The trained model files are excluded from the repository due to size constraints. You have two options:

### Option 1: Retrain Models Locally
Run the Jupyter notebook to train all models:
```bash
jupyter notebook ensemble_edition.ipynb
```

### Option 2: Download Pre-trained Models
Models will be available in a future release. Check the [Releases](https://github.com/iam-mukeshvenkatesh/Sentiment-analysis-on-Covid-19-Twitter-dataset/releases) section.

---

## Running the Web App

Once models are trained (or downloaded), start the Streamlit web app:

```bash
streamlit run GUI_Interface.py
```

Open `http://localhost:8501` in your browser. The app accepts any text input and returns a sentiment prediction with confidence scores for all three classes.

> **Note:** The current GUI uses simulated predictions as a demo. To enable real inference, ensure trained model files are in the root directory and the `load_models()` function in `GUI_Interface.py` is properly configured.

---

## Results

| Model | Accuracy |
|---|---|
| BERT-CNN-BiLSTM | ~93% |
| GloVe-CNN-BiLSTM | ~91% |
| SVM (GPU, linear) | **98.26%** |
| **Ensemble** | **94.95%** |

---

## Usage Example

```python
# Inference with the ensemble
from ensemble_edition import EnsemblePredictor

predictor = EnsemblePredictor()
text = "COVID-19 vaccines are now available"
sentiment, confidence = predictor.predict(text)
print(f"Sentiment: {sentiment}, Confidence: {confidence:.2f}")
```

---

## Technical Details

- **Framework**: PyTorch
- **NLP Library**: HuggingFace Transformers
- **GPU Support**: NVIDIA RAPIDS (cuML) for accelerated SVM training
- **Data Handling**: Pandas, NumPy
- **Visualization**: Streamlit

---

## Requirements

See `requirements.txt` for the complete list of dependencies and versions.

---

## Acknowledgements

- **Dataset**: [Arunavak Chakraborty on Kaggle](https://www.kaggle.com/datasets/arunavakrchakraborty/covid19-twitter-dataset)
- **BERT**: [HuggingFace Transformers](https://huggingface.co/bert-base-uncased)
- **GPU acceleration**: [NVIDIA RAPIDS (cuML)](https://rapids.ai/)

---

## License

This project is provided as-is for educational and research purposes. Please cite the original dataset authors if used in research.

---

## Contributing

Contributions are welcome! Feel free to:
- Report issues
- Suggest improvements
- Submit pull requests

---

## Author

**Mukesh Venkatesh** — [@iam-mukeshvenkatesh](https://github.com/iam-mukeshvenkatesh)
