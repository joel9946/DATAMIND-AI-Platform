# ⬡ DataMind — AI-Powered Data Science Platform

> End-to-end data science platform powered by local LLMs via Ollama.  
> No cloud. No API keys. No cost. 100% private.

![Python](https://img.shields.io/badge/Python-3.11+-blue?style=flat&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red?style=flat&logo=streamlit)
![Ollama](https://img.shields.io/badge/Ollama-Local_LLM-black?style=flat)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)

---

## ✨ Features

| Module | Status | Description |
|---|---|---|
| 🔬 EDA Lab | ✅ Live | Auto EDA — charts, correlations, outlier detection, AI narrative |
| 💬 Data Chat | ✅ Live | Ask questions in plain English → Pandas code → live results |
| 📚 RAG Studio | ✅ Live | Upload docs → vector search → grounded AI answers |
| 🤖 AutoML Arena | 🚧 In Development | 9 models race simultaneously → leaderboard → AI explanation |

---

## 🚀 Quick Start

**1. Clone the repo**
```bash
git clone https://github.com/joel9946/DATAMIND-AI-Platform
cd DataMind
```

**2. Create a virtual environment**
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Install and start Ollama**  
Download from [ollama.com](https://ollama.com), then:
```bash
ollama pull llama3.2
ollama serve
```

**5. Run the app**
```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`

---

## 🛠 Tech Stack

- **Frontend:** Streamlit, custom CSS (parallax, dark theme, animations)
- **AI/LLM:** Ollama (local), streaming token generation
- **ML:** scikit-learn (9 models), Pandas, NumPy
- **Visualization:** Plotly (interactive dark-theme charts)
- **RAG:** Custom vector store, character n-gram embeddings, cosine similarity
- **Data formats:** CSV, Excel, Parquet, JSON, PDF, TXT, DOCX

---

## 📁 Project Structure
```
DataMind/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── core/
│   ├── ollama_client.py    # Ollama API client + AI agents
│   ├── eda_engine.py       # Automated EDA pipeline
│   ├── automl_engine.py    # AutoML training (🚧 in dev)
│   └── rag_engine.py       # RAG pipeline + vector store
└── ui/
    └── styles.py           # CSS, animations, HTML components
```

---

## 🧠 How RAG Works (Built From Scratch)

1. Upload a document → text extracted
2. Text cut into overlapping 600-char chunks
3. Each chunk converted to a 128-dim vector (character n-gram hashing)
4. Vectors stored in an in-memory similarity search store
5. Your question → vectorised → top-5 most similar chunks retrieved
6. Chunks + question fused into a structured prompt → Ollama answers

---

## 🤖 AutoML Arena (Coming Soon)

Trains 9 models simultaneously with 5-fold cross-validation:
`Logistic Regression · Random Forest · Gradient Boosting · Extra Trees · Decision Tree · KNN · Naive Bayes · SVM · Linear Regression`

---

## 📄 License
MIT License — free to use, modify, and share.