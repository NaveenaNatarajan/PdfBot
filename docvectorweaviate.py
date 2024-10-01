import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import Weaviate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import docvectorweaviate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
os.getenv("MYAPI_KEY")
genai.configure(api_key=os.getenv("MYAPI_KEY"))

# Initialize Weaviate client
client = docvectorweaviate.Client(" ")   

 
def init_weaviate_schema():
    schema = {
        "classes": [
            {
                "class": "PDFTextChunk",
                "description": "Stores text chunks from PDFs",
                "properties": [
                    {
                        "name": "text",
                        "dataType": ["text"],
                        "description": "Text chunk"
                    },
                    {
                        "name": "embedding",
                        "dataType": ["number[]"],
                        "description": "Vector representation of the text"
                    }
                ],
                "vectorIndexType": "hnsw"
            }
        ]
    }
    
    if not client.schema.exists("PDFTextChunk"):
        client.schema.create(schema)

# PDF text extraction function
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Text chunking function
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Weaviate vector store creation
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    init_weaviate_schema()  # Initialize Weaviate schema
    vector_store = Weaviate.from_texts(text_chunks, embedding=embeddings, weaviate_client=client, class_name="PDFTextChunk")
    # No need to save locally as Weaviate handles the storage in its instance

# Conversational chain setup
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details,
    if the answer is not in provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n Question: \n{question}\n Answer: """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# User input handling function
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    # Load vectors from Weaviate
    new_db = Weaviate(weaviate_client=client, class_name="PDFTextChunk", embedding=embeddings)
    
    # Search for similar documents in Weaviate
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    st.write("Reply: ", response["output_text"])

# Streamlit app
def main():
    st.set_page_config("PDFBot")
    st.header("Chat with PDFs")

    user_question = st.text_input("Ask your question regarding the document!")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload and Click on the Submit & Proceed Button", accept_multiple_files=True)
        if st.button("Submit & Proceed"):
            with st.spinner("Generating..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")

if __name__ == "__main__":
    main()
