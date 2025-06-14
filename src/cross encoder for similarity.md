🤖 Best Cross-Encoder for Sentence Pair Classification
This project focuses on detecting the relationship between sentence pairs using a fine-tuned Cross-Encoder model. The architecture takes both sentences as a single input and directly predicts a similarity label — achieving high accuracy for tasks like paraphrase detection and semantic matching.
We use a state-of-the-art transformer-based DeBERTa-v3-Large model fine-tuned on a custom binary classification dataset.

🧠 Model Used
nli-deberta-v3-large (Cross-Encoder)
Source: cross-encoder/nli-deberta-v3-large from Hugging Face
Task: Binary Classification
0 = Not Duplicate
1 = Duplicate / Semantically Similar
Fine-tuned on a labeled dataset with sentence pairs (question1, question2)
Optimized using:
AdamW optimizer
Linear warm-up scheduler
CrossEntropyLoss

📊 Results
✅ Validation Accuracy: ~88–90%
📉 Loss: Rapid convergence after few epochs
🧪 Evaluated using:
Accuracy
Precision, Recall, F1-score
Confusion matrix

🔍 How It Works
Cross-Encoder inputs both sentences together:
"[CLS] Sentence1 [SEP] Sentence2 [SEP]"
Unlike bi-encoders, it does not embed sentences separately — leading to much better accuracy on fine-grained classification tasks.

📈 Why Cross-Encoder?
✅ Learns richer interaction between sentence pairs
⚠️ Slower inference than bi-encoder (not ideal for real-time bulk retrieval)
🎯 Ideal for high-accuracy tasks like duplicate detection, NLI, QA reranking

🧪 Future Improvements
Add early stopping and model checkpointing
Hyperparameter tuning (batch size, learning rate)
Export to ONNX for deployment

📝 License
This project is open for research and educational purposes. You may adapt it for your academic or personal use.



