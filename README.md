
# Guarded Exam – AI-Powered Plagiarism Detection & Auto-Grading<img align="right" width="161" height="157" top="-2px" src="https://github.com/user-attachments/assets/3e747d37-7668-44e5-b819-9eb14780f669">

**Guarded Exam** is a web-based exam submission system designed to enhance academic integrity using advanced Natural Language Processing (NLP). It automatically detects AI-generated answers and grades student submissions based on semantic similarity to model answers—without intrusive monitoring.

---
### 🚀 Features

- ✍️ **Exam Form Interface** – Simple, secure web form for student answers.
- 🤖 **AI Detection Module** – Identifies GPT-like AI-written content using fine-tuned transformer models.
- 🧠 **Similarity Scoring** – Automatically grades answers based on semantic similarity to tutor-provided answers.
- 📊 **Instructor Dashboard** – View results, detection outcomes, and grades.
- 🐳 **Dockerized Deployment** – Fully containerized via Docker and Kubernetes for easy scaling and reproducibility.

---

### 🧱 Architecture

- **Backend**: Django
- **Frontend**: HTML/CSS  
- **Models**:
  - `RoBERTa-large`, `GPT-2` (AI Detection)
  - `DeBERTa-v3-zero-shot`, `DeBERTa Cross-Encoder` (Similarity Scoring)
- **Database**: SQLite
- **Deployment**: Docker + KinD (Kubernetes in Docker)

---

### 🛠️ Installation

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

## 🧪 Testing & Results

| Module          | Metric        | Score     |
|----------------|---------------|-----------|
| AI Detection    | Accuracy      | 98.8%     |
| Similarity Model| Accuracy      | 93.0%     |
| False Positives | AI Detection  | < 3%      |
| False Positives | Similarity Model  | < 10%      |

All models were fine-tuned on large-scale datasets such as **AI vs Human Text**, **Quora Question Pairs**, and **SciTLDR**.

---

### 📦 Datasets Used

- AI vs Human Text (Kaggle)
- DAIGT-v2 Dataset
- Human vs LLM Text Corpus
- Quora Question Pairs
- STSB Multi-MT
- AllenAI SciTLDR

---

### 📜 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for more details.

---

### 🙏 Acknowledgements

- Hugging Face Transformers
- Django REST Framework
- Docker & Kubernetes community
- Our supervisors at FCAI-HU

### 📷 Images
  - Front Page:
    ![image](https://github.com/user-attachments/assets/2a8a1b41-e738-4037-a2d8-f5c6fbb54f85)
  - Admin View:
    ![image](https://github.com/user-attachments/assets/ee732e0c-d46a-4be4-87fd-42cb8545fd5e)
  - Teacher View:
    ![image](https://github.com/user-attachments/assets/ab3c8446-0edf-4b7f-bdce-0c9668b89560)
    ![image](https://github.com/user-attachments/assets/23229b99-2efd-465f-a316-1e410ea05200)
    ![image](https://github.com/user-attachments/assets/e497e7ea-1de1-4b86-806c-5a3c3e805227)
    ![image](https://github.com/user-attachments/assets/2bbdf312-bbf8-4b04-b0bb-19c5778d979b)
  - Student View:
    ![image](https://github.com/user-attachments/assets/24d6bfb8-04a2-4183-a099-cc55fcd6ec85)
    ![image](https://github.com/user-attachments/assets/3dd1680e-0d5c-481d-bc7d-0ddd1ec2e2bf)

        
