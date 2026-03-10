# 🧠 DataMind — AI-Powered Data Science Platform

> End-to-end data science platform powered by local LLMs via Ollama.
> No cloud. No API keys. No cost. 100% private.

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![Ollama](https://img.shields.io/badge/Ollama-Local_LLM-black?style=flat)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)

---

## ✨ Features
- 🔍 Natural language → Pandas queries (NL→EDA)
- 📄 RAG-powered document Q&A
- 🤖 AutoML with 9 models
- 📊 EDA automation with Plotly charts
- 🔒 100% local — no data leaves your machine

---

## ⚙️ Architecture
```
User Input (NL Query / Document)
        ↓
Streamlit UI
        ↓
[RAG Pipeline] OR [NL→Pandas Engine] OR [AutoML Module]
        ↓
Ollama Local LLM (no API needed)
        ↓
Results + Visualizations
```

## 🛠️ Tech Stack
Python · Streamlit · Ollama · Scikit-learn · Plotly · Pandas

---

## ▶️ How to Run
```bash
git clone https://github.com/joel9946/DATAMIND-AI-Platform.git
cd DATAMIND-AI-Platform
pip install -r requirements.txt
# Make sure Ollama is running locally
streamlit run app.py
```

---

## 📁 Project Structure
```
DATAMIND-AI-Platform/
├── app.py
├── core/          ← AI/ML logic
├── ui/            ← Streamlit components  
├── requirements.txt
└── README.md
```