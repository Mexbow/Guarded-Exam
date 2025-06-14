# 🧠 Zero-Shot Text Classifier

This project focuses on **zero-shot text classification** using a custom fine-tuned transformer model.  
The model is capable of predicting the most suitable label for a given text without requiring task-specific training data.

Ideal for use cases such as:
- Intent Detection  
- Topic Classification  
- Content Moderation  
- Spam Filtering

---

## 🤖 Model Used

### Custom Zero-Shot Transformer Model
- **Base Model**: Transformer-based architecture (fine-tuned for NLI-style classification)
- **Fine-tuned on**: A custom dataset for general-purpose zero-shot classification
- **Prediction Mechanism**: Uses entailment-style evaluation to compare input text with candidate labels

---

## 📦 Download Fine-Tuned Model

| Model Weights | Download Link |
|---------------|----------------|
| Zero-Shot Model (`.pt` or `.pkl`) | [Download from Google Drive](https://drive.google.com/file/d/1NA0t6rNjGBs5c8WGPnM9AduRnJa8gmaW/view?usp=drive_link) |

> After downloading, place the model file in your project directory, e.g. `/models/`.

---

## ⚙️ How It Works

1. **Input**: A raw sentence or document.
2. **Labels**: A list of user-defined candidate labels.
3. The model converts the task into a Natural Language Inference (NLI) problem:
4. The model ranks the labels based on "entailment" probabilities.

---

## 📊 Results

- ✅ High accuracy on varied topics  
- 🧪 Adaptable to unseen domains  
- ⚙️ Easy to integrate in production pipelines  
- 🔄 No re-training needed for new label sets

---

## 📈 Why Zero-Shot?

- 🚫 **No labeled data** required  
- 🔁 **Dynamic labels** supported at inference time  
- 💡 Useful for **multi-domain** or **rapid-deployment** NLP systems

---

## 🧪 Future Improvements

- Extend to multilingual classification  
- Improve label phrasing with prompt engineering  
- Add few-shot fallback mechanism

---

## 📝 License

This project is provided for **research and educational purposes**.  
You may adapt and build upon it for academic or personal use.

---
