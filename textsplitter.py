import streamlit as st
import os
from PyPDF2 import PdfReader
import pdfplumber
import fitz  #pymupdf
from pdfminer.high_level import extract_text
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import nltk
import spacy
import re

# Load environment variables and set up API key
load_dotenv()
genai.configure(api_key=os.getenv("MYAPI_KEY"))

# PDF extraction functions for different libraries
def get_pdf_text_pypdf2(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_pdf_text_pdfplumber(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        with pdfplumber.open(pdf) as pdf_reader:
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
    return text

def get_pdf_text_pymupdf(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = fitz.open(pdf)
        for page in pdf_reader:
            text += page.get_text()
    return text

def get_pdf_text_pdfminer(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        text += extract_text(pdf)
    return text

# Function to dynamically choose PDF reader
def get_pdf_text(pdf_docs, method):
    if method == "PyPDF2":
        return get_pdf_text_pypdf2(pdf_docs)
    elif method == "pdfplumber":
        return get_pdf_text_pdfplumber(pdf_docs)
    elif method == "PyMuPDF":
        return get_pdf_text_pymupdf(pdf_docs)
    elif method == "pdfminer":
        return get_pdf_text_pdfminer(pdf_docs)
    else:
        return "Invalid method selected!"
    


#Text splitting with different methods
def split_text_nltk(text):
    nltk.download('punkt')
    from nltk.tokenize import sent_tokenize
    return sent_tokenize(text)

def split_text_spacy(text):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    return[sent.text for sent in doc.sents]

def split_text_re(text, pattern=r'\n+'):
    return re.split(pattern, text)

def get_text_chunks(text, method, chunk_size, chunk_overlap):
    if method == "RecursiveCharacterTextSplitter":
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        return text_splitter.split_text(text)
    elif method == "NLTK Sentence Splitter":
        return split_text_nltk(text)
    elif method == "spaCy Sentence Splitter":
        return split_text_spacy(text)
    elif method == "Regex Splitter":
        return split_text_re(text)
    else:
        return ["Invalid text splitting method selected!"]


# Vector store creation using FAISS
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Conversational chain setup
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details,
    if the answer is not in the provided context just say, 'answer is not available in the context',
    don't provide the wrong answer\n\nContext:\n {context}?\n Question: \n{question}\n Answer: """
    
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

# User question input handling
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    st.write("Reply: ", response["output_text"])

# Streamlit app
def main():
    st.set_page_config("PDFBot")
    st.header("Chat with PDFs")

    # Dropdown to select the PDF extraction method
    method = st.sidebar.selectbox("Select PDF Extraction Method:", ["PyPDF2", "pdfplumber", "PyMuPDF", "pdfminer"])

    user_question = st.text_input("Ask your question regarding the document!")
    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload and Click on the Submit & Proceed Button", accept_multiple_files=True)
        if st.button("Submit & Proceed"):
            with st.spinner("Generating..."):
                raw_text = get_pdf_text(pdf_docs, method)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")

if __name__ == "__main__":
    main()
