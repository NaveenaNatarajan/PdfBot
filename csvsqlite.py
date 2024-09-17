import os
from dotenv import load_dotenv
import pandas as pd
import sqlite3
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import google.generativeai as genai
import streamlit as st

# Load environment variables
load_dotenv()
api_key = os.getenv("MYAPI_KEY")
genai.configure(api_key=api_key)

# Function to create SQLite database and store CSV data
def store_csv_in_db(csv_files):
    conn = sqlite3.connect('data.db')
    cursor = conn.cursor()
    
    # Create a new table for the data
    cursor.execute("DROP TABLE IF EXISTS csv_data")
    cursor.execute("""
    CREATE TABLE csv_data (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        content TEXT
    )
    """)
    
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        df['content'] = df.apply(lambda row: ' '.join(row.astype(str)), axis=1)
        df[['content']].to_sql('csv_data', conn, if_exists='append', index=False)
    
    conn.commit()
    conn.close()

# Function to retrieve text from SQLite database
def get_text_from_db():
    conn = sqlite3.connect('data.db')
    query = "SELECT content FROM csv_data"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return ' '.join(df['content'].tolist())

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create and save vector store
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Function to create conversational chain
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n Question: \n{question}\n Answer: """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Function to handle user input
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    try:
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    except ValueError as e:
        st.error(f"Error loading FAISS index: {e}")
        return
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    st.write("Reply: ", response["output_text"])

# Streamlit application
def main():
    st.set_page_config(page_title="CSVBot")
    st.header("Chat with CSVs")

    user_question = st.text_input("Ask your question regarding the document!")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        csv_files = st.file_uploader("Upload and Click on the Submit & Proceed Button", accept_multiple_files=True, type=["csv"])
        if st.button("Submit & Proceed"):
            with st.spinner("Generating..."):
                store_csv_in_db(csv_files)
                raw_text = get_text_from_db()
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")

if __name__ == "__main__":
    main()
