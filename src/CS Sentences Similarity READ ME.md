# 🔬 Scientific Sentence Similarity (SciTLDR + Cross-Encoder)

This project focuses on measuring semantic similarity between scientific sentences using a **Cross-Encoder architecture** built on top of **RoBERTa-Large**, fine-tuned on the [AllenAI SciTLDR dataset](https://huggingface.co/datasets/allenai/scitldr).

The objective is to predict a **ROUGE-L similarity score** between a source sentence and a scientific paper's TLDR-style summary sentence.

---

## 🧠 Model Used

### ✅ cross-encoder/stsb-roberta-large (Fine-tuned)

- **Base Model**: [`cross-encoder/stsb-roberta-large`](https://huggingface.co/cross-encoder/stsb-roberta-large)
- **Pretrained Task**: Sentence similarity on STS Benchmark (general domain)
- **Fine-tuned On**: [SciTLDR](https://huggingface.co/datasets/allenai/scitldr) dataset (scientific papers)
- **Task**: Regression (outputs a similarity score ∈ [0, 1])
- **Architecture**: Cross-encoder (both sentences passed together through the transformer)

---

## 📂 Dataset

### [AllenAI SciTLDR](https://huggingface.co/datasets/allenai/scitldr)

- **Domain**: Scientific papers (Computer Science, AI)
- **Structure**:
  - `source`: List of extracted sentences from a paper
  - `target`: Human-written TLDRs (single or multiple)
  - `rouge_scores`: ROUGE-L score between each source sentence and TLDR


The dataset was cleaned, filtered, and converted into sentence pairs suitable for regression training and evaluation.

The number of extracted trining exmaples for fine-tune after extraction is (Total pairs processed: 13236).

The evaluation examples are from the test split and total examples are 619.

---

## 🔗 Download Fine-Tuned Model & Notebook

Model and training notebook are available at the following Google Drive link:

📁 **[Download Fine-tuned Model Folder](https://drive.google.com/drive/folders/1cjlEGUWfSdFMzCa2E7YD6bhdCTJc8C07?usp=sharing)**

Place the extracted contents in a folder structure like:

/src/
├── model-4-isa.ipynb
├── Model4.rar
---

## 📊 Performance

| Metric                    | Value   |
|---------------------------|---------|
| **MSE Before Fine-Tuning** | 0.0896  |
| **MSE After Fine-Tuning**  | 0.0063  |
| **Δ MSE Improvement**      | 0.0832  |

- The model significantly improved after fine-tuning, learning domain-specific similarities in scientific text.
- **70% of RoBERTa layers** were frozen to preserve general knowledge while allowing domain adaptation.

---

## ⚙️ Training Strategy

- **Data Format**: `(sentence1, sentence2, label)` where label = ROUGE-L score
- **Cleaning**:
  - Removed HTML tags, URLs, and special characters
  - Dropped entries with nulls, misalignment, or empty summaries
- **Text Normalization**: lowercasing, trimming, and token filtering
- **Augmentation**: Random TLDR sampling when multiple TLDRs are present
- **Fine-Tuning Details**:
  - Optimizer: Adam
  - Learning Rate: 2e-5
  - Epochs: 15
  - Batch Size: 16
  - Warmup: 10% of total steps
- **Freezing Strategy**: 70% of RoBERTa base layers frozen (e.g., embedding + early layers)

---

## 📈 Why This Model?

- A **cross-encoder** is ideal for sentence-pair regression tasks as it computes attention across both sentences jointly.
- Compared to bi-encoders, cross-encoders offer **higher accuracy** for fine-grained similarity tasks (with increased compute cost at inference).

---

## 🔮 Future Work

- Use bi-encoder or dual-encoder architectures for faster inference with trade-offs
- Extend the task to multi-sentence or paragraph-level semantic similarity
- Evaluate generalization on other scientific datasets like S2ORC or arXiv summaries

---

## 📝 License

This project is open for **research and educational** purposes only. If using the fine-tuned model or methodology in your work, please cite appropriately.

---


### STSB-RoBERTa-Large Model

- [cross-encoder/stsb-roberta-large](https://huggingface.co/cross-encoder/stsb-roberta-large) by the [Sentence-Transformers team](https://www.sbert.net/)
- No specific paper, but based on RoBERTa ([Liu et al., 2019](https://arxiv.org/abs/1907.11692)) and STSB dataset

---

## 🧠 Authors

- 🔬 Soliman Khalil  
- 🤖 Powered by [Hugging Face](https://huggingface.co) and [Sentence Transformers](https://www.sbert.net/)

---

## 🛠 Requirements


pip install datasets pandas scikit-learn matplotlib sentence-transformers

---

### 🧪 Alternative Label Transformation (Under Evaluation)

This experiment explores a new distribution strategy for ROUGE similarity scores used as regression targets. Instead of using raw ROUGE scores, we apply a combination of **Yeo-Johnson transformation** followed by **Min-Max scaling** to normalize the labels to the range [0, 1].

---

## 📊 Why Use This Approach?

The transformation pipeline addresses several key issues with raw ROUGE scores:

- **Original ROUGE scores are skewed**, making them less ideal for regression tasks
- **Yeo-Johnson transformation** improves normality of the distribution
- **MinMaxScaler** ensures output aligns with the model's sigmoid activation (range [0, 1])

---

## 🧬 Transformation Pipeline

```python
from sklearn.preprocessing import PowerTransformer, MinMaxScaler

labels = df['label'].values.reshape(-1, 1)

# Step 1: Apply Yeo-Johnson transformation
pt = PowerTransformer(method='yeo-johnson', standardize=False)
labels_yeojohnson = pt.fit_transform(labels)

# Step 2: Normalize to [0, 1]
scaler = MinMaxScaler()
labels_scaled = scaler.fit_transform(labels_yeojohnson)

# Assign transformed values back to the DataFrame
df['label_transformed'] = labels_scaled
```

---

## ✅ Evaluation: Classification from Regression

Although the model is trained as a regressor, we evaluate its ability to distinguish relevant sentence pairs using a threshold-based classification approach.

### 🔹 Threshold Rule
- **Prediction > 0.6** → Positive Match
- **Prediction ≤ 0.6** → Negative Match

### 📈 Performance Metrics

```
=== Evaluation Metrics ===
Accuracy : 0.9167
Precision: 0.9412
Recall   : 0.8000
F1 Score : 0.8649
```

These results demonstrate **strong alignment** between the regression model's outputs and a binary classification perspective.

---

## 📂 Implementation Notes

- This approach is still **under evaluation**
- May be integrated in future versions if performance improvements hold consistently
- Results show promising potential for improving model performance on scientific text similarity tasks
