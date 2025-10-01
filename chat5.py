# AI Chatbot Assignment with Streamlit
# This single file contains the complete application for all three parts:
# Part A: Flow-Based Chatbot
# Part B: RAG (Retrieval-Augmented Generation) Chatbot
# Part C: A simple chat interface to switch between modes.

import streamlit as st
import re
import google.generativeai as genai
import os
import chromadb
from sentence_transformers import SentenceTransformer
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader

# --- Configuration ---
# To run this app, you need a Google API key for the Gemini model.
# 1. Get your API key from Google AI Studio: https://aistudio.google.com/app/apikey
# 2. Set it as an environment variable named GOOGLE_API_KEY.
# For Streamlit Community Cloud, you can set this in the app's secrets.
try:
    # Local development: Use st.secrets if available, otherwise get from environment
    if 'GOOGLE_API_KEY' in st.secrets:
        os.environ['GOOGLE_API_KEY'] = st.secrets['GOOGLE_API_KEY']
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    model = genai.GenerativeModel('gemini-2.5-flash')
except Exception as e:
    # If the API key is not configured, we will show an error message later.
    model = None

# --- Part A: Flow-Based Chatbot Logic ---
flow_questions = [
    {'key': 'name', 'prompt': 'Hello! I can help you with your order. What is your full name?', 'validation': r'.+'},
    {'key': 'email', 'prompt': 'Great, {name}! What is your email address?', 'validation': r'^[^\s@]+@[^\s@]+\.[^\s@]+$'},
    {'key': 'item', 'prompt': 'Thanks! What item are you interested in purchasing?', 'validation': r'.+'},
    {'key': 'quantity', 'prompt': 'How many units of "{item}" would you like?', 'validation': r'^[1-9]\d*$'}
]

def initialize_flow_state():
    """Initializes or resets the flow state in the session."""
    st.session_state.flow_state = {
        'current_question_index': 0,
        'user_data': {},
        'is_completed': False
    }
    # Ask the first question
    first_question = flow_questions[0]['prompt']
    st.session_state.flow_messages.append({"role": "assistant", "content": first_question})

def process_flow_input(user_input):
    """Processes user input for the flow-based chatbot."""
    fs = st.session_state.flow_state
    
    if fs['is_completed']:
        # response = "We've already completed the flow. To start over, please switch modes."
        st.chat_message("assistant").write("We've already completed the flow. To start over, please switch modes.")
        # st.chat_message("assistant").write(response)
        return

    current_question = flow_questions[fs['current_question_index']]

    # Validate user input
    if not re.match(current_question['validation'], user_input):
        # st.chat_message("assistant").write("Sorry, that doesn't look right. Could you please try again?")
        # st.session_state.messages.append({"role": "assistant", "content": response})
        # st.chat_message("assistant").write(response)
        response = "Sorry, that doesn't look right. Could you please try again?"
        st.session_state.flow_messages.append({"role": "assistant", "content": response})
        # st.chat_message("assistant").write(response)
        return
        # return

    # Save data and move to the next step
    fs['user_data'][current_question['key']] = user_input
    fs['current_question_index'] += 1

    if fs['current_question_index'] < len(flow_questions):
        # Ask the next question, personalizing it if needed
        next_question_template = flow_questions[fs['current_question_index']]['prompt']
        next_question = next_question_template.format(**fs['user_data'])
        st.session_state.flow_messages.append({"role": "assistant", "content": next_question})
        # st.chat_message("assistant").write(next_question)
    else:
        # End of flow, present summary
        fs['is_completed'] = True
        present_summary()

def present_summary():
    """Displays the final summary of the flow."""
    user_data = st.session_state.flow_state['user_data']
    summary = f"""
    ### Order Summary
    **Name:** {user_data['name']}  
    **Email:** {user_data['email']}  
    **Item:** {user_data['item']}  
    **Quantity:** {user_data['quantity']}
    
    A confirmation will be sent to your email. Thank you for your order!
    """
    st.session_state.flow_messages.append({"role": "assistant", "content": summary})
    
    follow_up_message = "To start a new order, please switch modes and then switch back to Flow Mode."
    st.session_state.flow_messages.append({"role": "assistant", "content": follow_up_message})
    # The rerun will handle displaying these.
    # st.chat_message("assistant").markdown(summary)
    # st.chat_message("assistant").write("To start a new order, please switch modes and then switch back to Flow Mode.")


# --- Part B: RAG (Retrieval-Augmented Generation) Logic ---
# Load the PDF
pdf_path = "mars.pdf"
pdf_reader = PdfReader(pdf_path)

# Extract text
raw_text = ""
for page in pdf_reader.pages:
    raw_text += page.extract_text() + "\n"

# Split into manageable chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,   # adjust size based on your use case
    chunk_overlap=50
)
chunks = splitter.split_text(raw_text)

# Initialize Chroma client
chroma_client = chromadb.PersistentClient(path="chroma_db")

# Create or get collection
collection = chroma_client.get_or_create_collection("mars_pdf")

# Load embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Generate embeddings
embeddings = embedder.encode(chunks).tolist()

# Add to Chroma
collection.add(
    documents=chunks,
    embeddings=embeddings,
    ids=[f"doc_{i}" for i in range(len(chunks))]
)
# 4. Retrieval function
def vector_retrieve(query, top_k=3):
    query_embedding = embedder.encode([query]).tolist()[0]
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    return results["documents"][0]

def process_rag_input(user_input):
    """Processes user input for the RAG chatbot."""

    # Store user query in history
    # st.session_state.rag_messages.append({"role": "user", "content": user_input})

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # 1. Retrieval
            relevant_contexts = vector_retrieve(user_input, top_k=3)
            relevant_context = "\n\n".join(relevant_contexts)


            if not relevant_context:
                response = "I'm sorry, I couldn't find any relevant information in my documents about that."
                st.session_state.rag_messages.append({"role": "assistant", "content": response})
                return

            # 2. Generation
            system_prompt = (
                "You are a helpful science tutor. Based ONLY on the provided context, "
                "answer the user's question. If the answer is not in the context, say that you "
                "cannot find the information in the provided documents."
            )
            generation_prompt = f"Context:\n{relevant_context}\n\nQuestion: {user_input}"

            try:
                response = model.generate_content([system_prompt, generation_prompt])
                answer = response.text
            except Exception as e:
                answer = f"⚠️ An error occurred while generating a response: {e}"

            # Save assistant answer in history
            st.session_state.rag_messages.append({"role": "assistant", "content": answer})


# --- Part C: Chat Interface ---
st.set_page_config(page_title="Multi-Mode AI Chatbot", layout="centered")

st.title("Multi-Mode AI Chatbot")
st.caption("A demo of flow-based and RAG-based chatbots")
# Add this at the top of your script (before sidebar code)
st.markdown(
    """
    <style>
    /* Sidebar background */
    [data-testid="stSidebar"] {
        background-color:  #B0E0E6; /* dark blue */
        color: white;
    }

    /* Sidebar header text */
    [data-testid="stSidebar"] h2, [data-testid="stSidebar"] label {
        color: black !important;
    }

    /* Radio buttons (labels) */
    div[role="radiogroup"] > label {
        color: white !important;
        font-weight: 500;
        padding: 4px 8px;
        border-radius: 6px;
        cursor: pointer;
    }

    /* Radio buttons hover effect */
    div[role="radiogroup"] > label:hover {
        background-color: #F5FBFF; /* slightly lighter blue */
    }

    /* Selected radio button */
    div[role="radiogroup"] > label[data-selected="true"] {
        background-color: #415a77; /* highlight color */
        color: #ffffff !important;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True
)
# Mode selection in the sidebar
with st.sidebar:
    st.header("Chat Mode")
    # Using radio buttons for mode selection
    new_mode = st.radio("Choose a mode:", ("Flow Mode", "RAG Mode"), label_visibility="collapsed")

# Initialize per-mode histories
if "rag_messages" not in st.session_state:
    st.session_state.rag_messages = []
if "flow_messages" not in st.session_state:
    st.session_state.flow_messages = []

# Initialize mode state
if "mode" not in st.session_state:
    st.session_state.mode = "Flow Mode"
    initialize_flow_state()

# Handle mode switching
if new_mode != st.session_state.mode:
    st.session_state.mode = new_mode

    if st.session_state.mode == "Flow Mode":
        # Reset flow history + state
        st.session_state.flow_messages = []
        initialize_flow_state()
        
    else:  # RAG Mode
        # Reset rag history + state
        st.session_state.rag_messages = []
        st.session_state.rag_state = {
            "user_data": {},
            "is_completed": False
        }
        st.session_state.rag_messages.append({
            "role": "assistant",
            "content": "RAG Mode is active. You can now ask questions about the planet Mars. For example: 'How are dust storms formed in Mars? or What is Borealis basin'"
        })

# Pick active history based on mode
if st.session_state.mode == "RAG Mode":
    active_history = st.session_state.rag_messages
else:
    active_history = st.session_state.flow_messages

# 🔄 Replay chat every rerun (only active mode)
for msg in active_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Main chat input and logic
if prompt := st.chat_input("Type your message..."):
    # Add user message to active history
    active_history.append({"role": "user", "content": prompt})

    if model is None:
        st.error("Google API Key is not configured. Please set the GOOGLE_API_KEY environment variable.")
    else:
        if st.session_state.mode == "Flow Mode":
            process_flow_input(prompt)
        else:  # RAG Mode
            process_rag_input(prompt)

        # Rerun so new messages display immediately
        st.rerun()
