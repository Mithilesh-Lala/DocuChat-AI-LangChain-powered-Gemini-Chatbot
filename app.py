import streamlit as st
import google.generativeai as genai
from PyPDF2 import PdfReader
import docx
import pandas as pd
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms.base import LLM
from typing import Any, List, Optional

# Configure Gemini API
genai.configure(api_key='Your-API-KEY-HERE')

# Initialize the Gemini model
gemini_model = genai.GenerativeModel('gemini-pro')

# Custom LLM class for Gemini
class GeminiLLM(LLM):
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        response = gemini_model.generate_content(prompt)
        return response.text

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"name": "GeminiLLM"}

    @property
    def _llm_type(self) -> str:
        return "GeminiLLM"

# Initialize HuggingFace embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Initialize FAISS vector store
vector_store = FAISS.from_texts([""], embeddings)

# Initialize conversation memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

def extract_text_from_pdf(file):
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def extract_text_from_docx(file):
    doc = docx.Document(file)
    text = ""
    for para in doc.paragraphs:
        text += para.text + "\n"
    return text

def extract_text_from_excel(file):
    df = pd.read_excel(file)
    return df.to_string()

def extract_text_from_json(file):
    data = json.load(file)
    return json.dumps(data, indent=2)

def process_uploaded_file(file):
    if file.type == "application/pdf":
        text = extract_text_from_pdf(file)
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        text = extract_text_from_docx(file)
    elif file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
        text = extract_text_from_excel(file)
    elif file.type == "application/json":
        text = extract_text_from_json(file)
    elif file.type == "text/plain":
        text = file.getvalue().decode("utf-8")
    else:
        st.error(f"Unsupported file type: {file.type}")
        return

    # Split the text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)

    # Add chunks to the vector store
    global vector_store
    vector_store = FAISS.from_texts(chunks, embeddings)

    st.success("File processed and added to the knowledge base.")

def main():
    st.set_page_config(layout="wide")

    # Sidebar for document upload
    with st.sidebar:
        st.title("Document Upload")
        uploaded_file = st.file_uploader("Upload a document (PDF, DOCX, Excel, JSON, or TXT)",
                                         type=["pdf", "docx", "xlsx", "json", "txt"])
        if uploaded_file is not None:
            process_uploaded_file(uploaded_file)

    # Main chat interface
    st.title("DocuChat AI: LangChain-powered Gemini Chatbot")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("What is your question?"):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Create a ConversationalRetrievalChain with GeminiLLM
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=GeminiLLM(),
            retriever=vector_store.as_retriever(),
            memory=memory
        )

        # Get response from the chain
        result = qa_chain({"question": prompt})
        response = result['answer']

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()