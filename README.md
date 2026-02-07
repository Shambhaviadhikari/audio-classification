## Animal Sound Classifier

This project is for **CSCI 6366: Neural Networks & Deep Learning** at The George Washington University.

---

## **FINAL SUBMISSION NOTEBOOK**

 `FINAL_project_submission.ipynb` 

**This is our final, comprehensive submission notebook that consolidates all our work:**

- Complete Exploratory Data Analysis (EDA)
- Baseline CNN implementation and results
- Full dataset CNN experiments with regularization
- Transfer learning with YAMNet embeddings
- Comprehensive metrics (Accuracy, Precision, Recall, F1-score) for all models on both validation and test sets
- Complete visualizations, confusion matrices, and training curves
- Detailed analysis, key findings, and conclusions


## Project Overview

We build deep learning models to classify animal sounds (**dog**, **cat**, **bird**) from short audio clips using:

- Mel-spectrograms and 2D Convolutional Neural Networks (CNNs)
- Hybrid **CRNN** architectures (CNN + GRU)
- Sequence models based on **Vision Transformers (ViT)** over spectrogram "images"
- Transfer learning with pre-trained audio models such as **YAMNet**

The end goal is to build a clean, reproducible pipeline and compare a simple CNN baseline against more advanced architectures and transfer-learning–based approaches.



**Key Results:**

- **92% test accuracy** with CNN + Dropout
- **Balanced performance** across all three classes
- **Research insight:** Task-specific training outperformed transfer learning by 26 percentage points

---

## Team Members

- Shambhavi Adhikari (G37903602) — GitHub: `@Shambhaviadhikari`
- Rakshitha Mamilla (G23922354) — GitHub: `@M-Rakshitha`
- Abhiyan Sainju (G22510509) — GitHub: `@aabhiyann`

---

## Dataset

We use the **Human Words Audio Classification** dataset (Kaggle):

https://www.kaggle.com/datasets/chiragchhaya/human-words-audio-classification

Each audio file is labeled as:

- `dog`
- `cat`
- `bird`

Properties:

- Mono `.wav` file
- Automatically resampled to **16 kHz**
- ~1 second duration
- Converted into **128×128 Mel-spectrograms**

---

## Project Structure

- **`FINAL_project_submission.ipynb`** **MAIN SUBMISSION**

  - Located in both the root directory and `notebooks/` folder
  - **This is the final, comprehensive notebook containing all our work**
  - Includes EDA, all model implementations, comprehensive metrics, visualizations, and conclusions
  - **Please use this notebook for evaluation**

- `data/`

  - `dog/` – WAV files labeled as dog
  - `cat/` – WAV files labeled as cat
  - `bird/` – WAV files labeled as bird


- `README.md` – this file

---

## Environment & Setup

- **Python**: 3.10+ recommended
- **Key packages**:
  - `tensorflow>=2.16,<3`
  - `librosa`
  - `numpy`
  - `matplotlib`
  - `scikit-learn`
  - `tensorflow_hub` (for YAMNet transfer learning)

Install dependencies (example with `pip`):

```bash
pip install "tensorflow>=2.16,<3" librosa numpy matplotlib scikit-learn tensorflow_hub
```

Place the dataset under `data/` with subfolders `dog/`, `cat/`, and `bird/` so that paths look like:

```text
audio-classification-cnn/
  data/
    dog/*.wav
    cat/*.wav
    bird/*.wav
```

Then you can open the notebooks in Jupyter or VS Code and run them end-to-end.

---


## Overall Comparison

| Model                  |      Metric Split | Accuracy | Precision |   Recall | F1-Score | Test Loss | Notes                                       |
| ---------------------- | ----------------: | -------: | --------: | -------: | -------: | --------: | ------------------------------------------- |
| Baseline CNN           |  Test (full data) |     ~90% |      ~90% |     ~90% |     ~90% |     ~0.57 | Trained from scratch                        |
| **CNN + Dropout(0.5)** |  Test (full data) | **~92%** |  **~92%** | **~92%** | **~92%** | **~0.24** | **Best model**                              |
| Baseline CNN           |   Val (full data) |     ~94% |      ~94% |     ~94% |     ~94% |       N/A | Trained from scratch                        |
| **CNN + Dropout(0.5)** |   Val (full data) | **~95%** |  **~95%** | **~95%** | **~95%** |       N/A | **Best model**                              |
| YAMNet (Full Sequence) |  Test (full data) |     ~66% |      ~60% |     ~58% |     ~58% |     ~0.96 | Transfer learning - preserves temporal info |
| YAMNet (Full Sequence) | Train (full data) |     ~87% |      ~87% |     ~87% |     ~87% |       N/A | Transfer learning - training metrics        |
| CRNN (CNN + BiGRU)     | Val (80/20 split) |  ~78.69% |       N/A |      N/A |      N/A |     ~0.80 | Validation metrics only                     |
| YAMNet (Averaged)      |  Test (full data) |     ~62% |       N/A |      N/A |      N/A |     ~0.90 | Transfer learning from AudioSet             |
| ViT-style Transformer  | Val (80/20 split) |  ~35–40% |       N/A |      N/A |      N/A |     ~1.10 | Validation metrics only                     |


