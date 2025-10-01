# Multimode AI Chatbot

This project is a **Streamlit-based multimode AI chatbot** that supports two modes:
1. **Flow Mode** – A standard conversational mode using LLMs.
2. **RAG Mode** – A Retrieval-Augmented Generation (RAG) mode that retrieves relevant information from custom documents (like PDFs) using **ChromaDB** and **Sentence Transformers**.

---

## 🚀 Features
- Interactive chat UI built with **Streamlit**.
- Mode switching between **Flow** and **RAG**.
- **PDF ingestion** with text splitting using `langchain-text-splitters`.
- Vector embeddings using `sentence-transformers`.
- Vector storage & retrieval using **ChromaDB**.
- LLM integration with **Google Generative AI**.
- Session history to maintain context.

---

## 📂 Project Structure
```
├── app.py                 # Main Streamlit app
├── requirements.txt       # Project dependencies
├── README.md              # Project documentation
├── chroma_db/             # Local vector database (auto-created by Chroma)
└── .streamlit/
    └── secrets.toml       # API keys (not pushed to GitHub)
```

---

## 🔑 Setup Instructions

### 1. Clone Repository
```bash
git clone https://github.com/your-username/multimode-ai-chatbot.git
cd multimode-ai-chatbot
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate   # On Mac/Linux
venv\Scripts\activate    # On Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Add API Keys
Create a `.streamlit/secrets.toml` file and add:
```toml
[general]
GOOGLE_API_KEY = "your_google_api_key_here"
```

### 5. Run the App
```bash
streamlit run app.py
```

---

## ⚠️ Notes
- Do **NOT** commit `.streamlit/secrets.toml` to GitHub.
---

## 📌 Future Improvements
- Support for more LLM providers (OpenAI, Anthropic, etc.).
- Enhanced document handling (DOCX, TXT, multiple PDFs).
- UI improvements with custom themes.
