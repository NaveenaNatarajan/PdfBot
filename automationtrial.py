import os
from dotenv import load_dotenv
import pandas as pd
from pandas_profiling import ProfileReport  # Import pandas-profiling
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import google.generativeai as genai
import streamlit as st

# API key loading
load_dotenv()
api_key = os.getenv("MYAPI_KEY")
genai.configure(api_key=api_key)

#-------------------------------------------------------
# Function to extract text from CSV and generate a profiling report
def get_csv_text_and_report(csv_files):
    combined_text = ""
    combined_df = pd.DataFrame()  # To store all CSV data
    
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        combined_text += df.to_string(index=False)
        combined_df = pd.concat([combined_df, df], ignore_index=True)  # Combine all data into one DataFrame
    
    # Generate profiling report
    profile = ProfileReport(combined_df, title="Pandas Profiling Report", minimal=True)
    profile.to_file("report.html")  # Save report for later use

    # Extract some key insights from the report and return with the text
    report_summary = profile.get_description()
    key_insights = f"Profiling Report Summary: \n{report_summary['table']['n']} rows, {report_summary['table']['n_var']} variables."
    
    # Combine the CSV data with the profiling insights
    combined_text += "\n\n" + key_insights
    return combined_text
#-------------------------------------------------------

# Split text into chunks for vector embedding
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Vector store for storing document embeddings
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Conversational chain for the question-answering system
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in provided context just say, "answer is not available in the context", don't provide the wrong answer.\n\n
    Context:\n {context}?\n Question: \n{question}\n Answer: """
    
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Function to process user input and query the FAISS index
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    # Load the FAISS index and perform a similarity search
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    st.write("Reply: ", response["output_text"])

# Streamlit app
def main():
    st.set_page_config(page_title="CSVBot with Profiling")
    st.header("Chat with CSVs and Profiling Report")

    user_question = st.text_input("Ask your question regarding the document!")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        csv_files = st.file_uploader("Upload and Click on the Submit & Proceed Button", accept_multiple_files=True, type=["csv"])
        
        if st.button("Submit & Proceed"):
            with st.spinner("Processing..."):
                # Get text from the CSV and generate a profiling report
                raw_text = get_csv_text_and_report(csv_files)
                
                # Split the combined text into chunks
                text_chunks = get_text_chunks(raw_text)
                
                # Create a vector store from the text chunks
                get_vector_store(text_chunks)
                
                st.success("CSV processed and ready for querying!")
                st.write("You can view the profiling report [here](report.html)")

if __name__ == "__main__":
    main()

