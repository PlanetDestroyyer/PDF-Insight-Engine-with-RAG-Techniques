import streamlit as st
import os
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama

st.set_page_config(page_title="QA_BOT", layout="wide", initial_sidebar_state="auto", menu_items=None)

llm = Ollama(model="llama3")

pdf_paths = [
    'uber-10-k-2023.pdf',
    'tsla-20231231-gen.pdf',
    'goog-10-k-2023 (1).pdf'
]

def get_ans(pdfs, question):
    all_chunks = []
    for pdf_path in pdf_paths:
        loader = UnstructuredPDFLoader(pdf_path)
        documents = loader.load()
        text_splitter = CharacterTextSplitter(
            separator='\n',
            chunk_size=3500,
            chunk_overlap=1000
        )
        text_chunks = text_splitter.split_documents(documents)
        all_chunks.extend(text_chunks)
    embeddings = HuggingFaceEmbeddings()
    knowledge_base = FAISS.from_documents(all_chunks, embeddings)
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=knowledge_base.as_retriever()
    )

    response = qa_chain.invoke(question)
    return response['result']

st.title("QA Bot")
st.write("Ask questions about the provided PDF documents.")


if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


if prompt := st.chat_input("What is your question?"):

    st.chat_message("user").markdown(prompt)

    st.session_state.messages.append({"role": "user", "content": prompt})

    response = get_ans(pdf_paths, prompt)
    

    with st.chat_message("assistant"):
        st.markdown(response)
 
    st.session_state.messages.append({"role": "assistant", "content": response})