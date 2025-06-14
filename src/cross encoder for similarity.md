# 🤖 Best Cross-Encoder for Sentence Pair Classification

This project focuses on detecting the relationship between sentence pairs using a fine-tuned **Cross-Encoder** model.  
The architecture takes both sentences as a single input and directly predicts a similarity label — achieving high accuracy for tasks like **paraphrase detection** and **semantic matching**.

We use a state-of-the-art transformer-based **DeBERTa-v3-Large** model fine-tuned on a custom binary classification dataset.

---

## 🧠 Model Used

### `nli-deberta-v3-large` (Cross-Encoder)
- **Source**: [`cross-encoder/nli-deberta-v3-large`](https://huggingface.co/cross-encoder/nli-deberta-v3-large) from Hugging Face  
- **Task**: Binary Classification  
  - `0 = Not Duplicate`  
  - `1 = Duplicate / Semantically Similar`
- **Fine-tuned on**: A labeled dataset with sentence pairs (`question1`, `question2`)
- **Optimized using**:
  - `AdamW` optimizer  
  - Linear warm-up scheduler  
  - `CrossEntropyLoss`

---

## 📦 Download Fine-Tuned Model

| Model Weights | Download Link |
|---------------|----------------|
| Cross-Encoder (`.pt` format) | [Download from Google Drive](https://drive.google.com/file/d/1GPF37eJZ7gGVVqOQ7XwGVduEBrXVOs75/view?usp=drive_link) |

> Place the downloaded weights file in your project folder under `/models/` or modify the notebook path accordingly.

---

## 📊 Results

- ✅ **Validation Accuracy**: ~88–90%  
- 📉 **Loss**: Rapid convergence after few epochs  
- 🧪 **Evaluation Metrics**:
  - Accuracy  
  - Precision, Recall, F1-score  
  - Confusion Matrix

---

## 🔍 How It Works

- The **Cross-Encoder** takes both sentences as a single input:
- Unlike bi-encoders, it does **not** embed sentences independently.  
This allows the model to learn **richer semantic interactions** between the inputs — improving performance on classification tasks.

---

## 📈 Why Cross-Encoder?

- ✅ Learns **deep relationships** between sentence pairs  
- ⚠️ **Slower inference** compared to bi-encoders  
- 🎯 Ideal for **high-accuracy** use cases like:
- Duplicate detection
- Natural Language Inference (NLI)
- QA reranking

---

## 🧪 Future Improvements

- ✅ Add **early stopping** and **model checkpointing**  
- 📊 Tune hyperparameters: learning rate, batch size, scheduler  
- 📦 Export the model to **ONNX** for faster deployment

---

## 📝 License

This project is open for **research and educational purposes**.  
You are free to use, modify, and adapt the code for academic or non-commercial applications.

---
