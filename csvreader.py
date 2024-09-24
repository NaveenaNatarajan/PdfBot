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
