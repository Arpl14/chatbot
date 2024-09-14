import os
import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from chromadb.config import Settings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate




# Load environment variables from .env file
load_dotenv()
os.getenv("GOOGLE_API_KEY")

import google.generativeai as genai
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Extract all of the texts from the PDF files 
def get_pdf_text(pdfs):
    text = ""
    for pdf in pdfs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text 

# Split text into chunks
def generate_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    chunks = text_splitter.split_text(text)
    return chunks 

# Convert Chunks into Vectors using Chroma
def chunks_to_vectors(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # Create a new Chroma index
    vector_store = Chroma(collection_name="document_embeddings", 
                          embedding_function=embeddings, 
                          persist_directory="chroma_index") 
    
    # Add the chunks to the vector store
    for chunk in chunks:
        vector_store.add_texts([chunk])

    # Save the index (optional, Chroma persists automatically if set to do so)
    vector_store.persist()

    return vector_store

# Get conversation chain using Google Generative AI
def get_conversation():
    prompt_template = """
    Answer the question that is asked with as much detail as you can, given the context that has been provided. If you are unable to come up with an answer based on the provided context,
    simply say "Answer cannot be provided based on the context that has been provided", instead of trying to forcibly provide an answer.\n\n
    Context: \n {context}?\n
    Question: \n {question}\n
    Answer:
    """
    
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# User input handling
def user_input(question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # Load the existing Chroma index
    vector_store = Chroma(collection_name="document_embeddings", 
                          embedding_function=embeddings, 
                          persist_directory="chroma_index")

    docs = vector_store.similarity_search(question)

    chain = get_conversation()

    response = chain({"input_documents": docs, "question": question}, return_only_outputs=True)
    st.write("Reply: ", response["output_text"])

# Main app function
def app():
    st.title("ASTM Documents Chatbot")
    st.sidebar.title("Upload Documents")

    # Sidebar for PDF file upload
    pdf_docs = st.sidebar.file_uploader("Upload your documents in PDF format, then click Analyze.", accept_multiple_files=True)

    analyze_triggered = st.sidebar.button("Chat Now")

    if analyze_triggered:
        with st.spinner("Configuring... ⏳"):
            raw_text = get_pdf_text(pdf_docs)
            chunks = generate_chunks(raw_text)
            chunks_to_vectors(chunks)
            st.success("Done")

    # User question input
    user_question = st.text_input("Ask a question based on the documents that were uploaded")

    if user_question:
        user_input(user_question)

if __name__ == "__main__":
    app()
