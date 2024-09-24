import os
from dotenv import load_dotenv
import pandas as pd
import csv
import dask.dataframe as dd
import pyarrow.csv as pv
from io import StringIO
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import google.generativeai as genai
import streamlit as st

# Load API key
load_dotenv()
api_key = os.getenv("MYAPI_KEY")
genai.configure(api_key=api_key)


#-------------------------------------------------------
# CSV extraction functions for different libraries

def read_csv_pandas(csv_files):
    text = ""
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        text += df.to_string(index=False)
    return text

def read_csv_python_csv(csv_files):
    text = ""
    for csv_file in csv_files:
        csv_file.seek(0)  # reset pointer
        content = csv_file.read().decode("utf-8")
        data = list(csv.reader(StringIO(content)))
        df = pd.DataFrame(data[1:], columns=data[0])
        text += df.to_string(index=False)
    return text

def read_csv_dask(csv_files):
    text = ""
    for csv_file in csv_files:
        df = dd.read_csv(csv_file).compute()
        text += df.to_string(index=False)
    return text

def read_csv_pyarrow(csv_files):
    text = ""
    for csv_file in csv_files:
        csv_file.seek(0)  # reset pointer
        table = pv.read_csv(csv_file)
        df = table.to_pandas()
        text += df.to_string(index=False)
    return text

# Function to dynamically choose CSV reader
def get_csv_text(csv_files, method):
    if method == "Pandas":
        return read_csv_pandas(csv_files)
    elif method == "Python CSV":
        return read_csv_python_csv(csv_files)
    elif method == "Dask":
        return read_csv_dask(csv_files)
    elif method == "PyArrow":
        return read_csv_pyarrow(csv_files)
    else:
        return "Invalid method selected!"
#----------------------------------------
# Text chunking
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)

# Vector store creation
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Conversational chain setup
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in the provided context, 
    just say 'answer is not available in the context', don't provide a wrong answer.\n\n
    Context:\n {context}\n Question: {question}\n Answer: """
    
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
    st.set_page_config(page_title="CSVBot")
    st.header("Chat with CSVs")

    # Dropdown to select the CSV extraction method
    method = st.sidebar.selectbox("Select CSV Reading Method:", ["Pandas", "Python CSV", "Dask", "PyArrow"])

    user_question = st.text_input("Ask your question regarding the document!")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        csv_files = st.file_uploader("Upload and Click on the Submit & Proceed Button", accept_multiple_files=True, type=["csv"])
        if st.button("Submit & Proceed"):
            with st.spinner("Processing..."):
                raw_text = get_csv_text(csv_files, method)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Ready to go!")

if __name__ == "__main__":
    main()