# Financial Sentiment Classification — Naive Bayes vs Logistic Regression

Classify financial news headlines and forum posts into **three sentiment classes**: **negative / neutral / positive**.  
This project implements **Naive Bayes** and **Logistic Regression**, compares their performance, and generates predictions for the **FiQA test set**.

> Project report/reference: `project summary.pdf`

---

## 1) Task Overview

- **Goal:** Sentiment classification for short financial text (headlines + forum posts)
- **Dataset:** **FiQA 2018 Task 1**
- **Labels (3-class):** negative / neutral / positive  
  The original dataset provides a **continuous sentiment score** per example. For supervised classification, we **map scores into 3 classes using quantiles** computed on training data:
  - lowest 33% → **negative**
  - middle 33% → **neutral**
  - highest 33% → **positive**

---

## 2) Data Processing

1. Combine the two FiQA training files
2. Split into:
   - **80% train**
   - **20% validation**
3. Build vocabulary **only from the training split**, using a **minimum frequency threshold**
4. Use the same vocabulary for validation + test
5. The FiQA test set has **no labels**, so it is used **only for generating predictions**

---

## 3) Models

### Naive Bayes (NB)
- Bag-of-words style features
- Fast baseline; tends to rely heavily on word frequency

### Logistic Regression (LR)
- Linear classifier over **TF–IDF** features
- Learns feature weights that better capture sentiment cues

---

## 4) Evaluation

**Metrics**
- Accuracy
- Macro-F1

**Main Results**
| Model | Train Acc | Train Macro-F1 | Val Acc | Val Macro-F1 |
|------|----------:|---------------:|--------:|-------------:|
| Naive Bayes | 0.9054 | 0.9054 | 0.5336 | 0.5311 |
| Logistic Regression | 0.9966 | 0.9966 | 0.6233 | 0.6210 |

**Confusion Matrices (Validation)**

Naive Bayes:
[43 16 14
21 32 21
17 15 44]

Logistic Regression:
[54 11  8
18 38 18
10 19 47]

**Per-class Accuracy (Validation)**
| Class | NB | LR |
|------|---:|---:|
| Negative | 0.589 | 0.740 |
| Neutral  | 0.432 | 0.514 |
| Positive | 0.579 | 0.618 |

**Strong vs Weak Sentiment (Validation Accuracy)**
(“strong/weak” is split by the absolute value of the original continuous score.)
| Model | Val Acc (strong) | Val Acc (weak) |
|------|------------------:|---------------:|
| Naive Bayes | 0.633 | 0.439 |
| Logistic Regression | 0.688 | 0.561 |

**Takeaway**
- **Logistic Regression performs better** on both validation accuracy and macro-F1.
- NB shows a larger train→val drop, indicating **overfitting**.
- **Neutral** is the hardest class for both models.

---

## 5) Test Predictions Output

The project generates predictions for the FiQA test set and saves them to a **CSV** file containing:
- numeric labels
- text labels (`negative`, `neutral`, `positive`)

Example (from report):
| id | sentence (truncated) | NB pred | LR pred |
|---:|---|---|---|
| 0 | "Cuadrilla ... delay application to frack ..." | neutral | neutral |
| 1001 | "Sainsbury ... warns of squeeze ..." | positive | neutral |
| 1006 | "Barclays ... fined for anti-money-laundering ..." | neutral | neutral |

---

## 6) Error Analysis (What’s Hard)

- Both models struggle most on **neutral**.
- Many neutral sentences contain words that also appear in positive/negative contexts.
- NB is more sensitive to frequent words and can be pushed to the wrong side.
- LR does better because TF–IDF + learned weights emphasize more sentiment-relevant cues.

---

## 7) Additional Experiments

- Tried **bigram features** and a simple **RNN** model
- No improvement, likely due to:
  - small dataset
  - short text length
  - limited benefit from more complex modeling in this setup