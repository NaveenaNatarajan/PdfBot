import streamlit as st
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
os.getenv("MYAPI_KEY")
genai.configure(api_key=os.getenv("MYAPI_KEY"))

# Extract text from PDFs using pdfplumber
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        with pdfplumber.open(pdf) as pdf_reader:
            for page in pdf_reader.pages:
                text += page.extract_text()
    return text

# Divide text into chunks (with customizable chunk size in Streamlit)
def get_text_chunks(text, chunk_size, chunk_overlap):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_text(text)
    return chunks

# Convert texts into vectors using FAISS
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Set up a conversational chain with Gemini
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details. 
    If the answer is not in the provided context just say, "answer is not available in the context", don't provide a wrong answer.\n\n
    Context:\n {context}?\n Question: \n{question}\n Answer: """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.7)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Get user input and connect with Gemini using the conversational chain
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    st.write("Reply: ", response["output_text"])

# Build the Streamlit application
def main():
    st.set_page_config("PDFBot")
    st.header("Chat with PDFs")

    user_question = st.text_input("Ask your question regarding the document!")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload and Click on the Submit & Proceed Button", accept_multiple_files=True)
        
        # Adding chunk size and overlap customization
        chunk_size = st.number_input("Enter chunk size (characters)", min_value=1000, max_value=20000, value=10000, step=500)
        chunk_overlap = st.number_input("Enter chunk overlap (characters)", min_value=0, max_value=5000, value=1000, step=100)

        if st.button("Submit & Proceed"):
            with st.spinner("Generating..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text, chunk_size, chunk_overlap)
                get_vector_store(text_chunks)
                st.success("Done")

if __name__ == "__main__":
    main()
