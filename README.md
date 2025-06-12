
# 🛡️ Guarded Exam – AI-Powered Academic Integrity Platform

**Guarded Exam** is a web-based exam submission system designed to enhance academic integrity using advanced Natural Language Processing (NLP). It automatically detects AI-generated answers and grades student submissions based on semantic similarity to model answers—without intrusive monitoring.

---

## 🚀 Features

- ✍️ **Exam Form Interface** – Simple, secure web form for student answers.
- 🤖 **AI Detection Module** – Identifies GPT-like AI-written content using fine-tuned transformer models.
- 🧠 **Similarity Scoring** – Automatically grades answers based on semantic similarity to tutor-provided answers.
- 📊 **Instructor Dashboard** – View results, detection outcomes, and grades.
- 🐳 **Dockerized Deployment** – Fully containerized via Docker and Kubernetes for easy scaling and reproducibility.

---

## 🧱 Architecture

- **Backend**: Django + Django REST Framework  
- **Frontend**: HTML/CSS  
- **Models**:
  - `RoBERTa-large`, `GPT-2` (AI Detection)
  - `DeBERTa-v3-zero-shot`, `DeBERTa Cross-Encoder` (Similarity Scoring)
- **Database**: PostgreSQL
- **Deployment**: Docker + KinD (Kubernetes in Docker)

---

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/guarded-exam.git
cd guarded-exam

# Build and start containers
docker-compose up --build
```

Or deploy via Kubernetes:

```bash
# Load Docker images into KinD
kind load docker-image guarded-exam-api

# Apply Kubernetes manifests
kubectl apply -f k8s/
```

---

## 📂 Project Structure

```
guarded-exam/
├── backend/             # Django backend
│   ├── ai_detection/    # AI detection logic (RoBERTa, GPT-2)
│   ├── similarity/      # Similarity scoring logic (DeBERTa)
│   └── api/             # REST API endpoints
├── frontend/            # HTML/CSS submission form
├── models/              # Saved model weights (optional)
├── docker-compose.yml   # Local container orchestration
├── k8s/                 # Kubernetes manifests
└── README.md
```

---

## 🧪 Testing & Results

| Module          | Metric        | Score     |
|----------------|---------------|-----------|
| AI Detection    | Accuracy      | 98.8%     |
| Similarity Model| F1-Score      | 89.7%     |
| False Positives | AI Detection  | < 2%      |

All models were fine-tuned on large-scale datasets such as **AI vs Human Text**, **Quora Question Pairs**, and **SciTLDR**.

---

## 📦 Datasets Used

- AI vs Human Text (Kaggle)
- DAIGT-v2 Dataset
- Human vs LLM Text Corpus
- Quora Question Pairs
- STSB Multi-MT
- AllenAI SciTLDR

---

## 📜 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for more details.

---

## 🙏 Acknowledgements

- Hugging Face Transformers
- Django REST Framework
- Docker & Kubernetes community
- Our supervisors at FCAI-HU
