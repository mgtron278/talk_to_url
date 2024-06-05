import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import requests
from bs4 import BeautifulSoup
import tempfile

google_api_key = os.environ.get("GOOGLE_API_KEY")
genai.configure(api_key=google_api_key)

def get_url_text(url):
    text=""
    response = requests.get(url)

    soup = BeautifulSoup(response.text, 'html.parser')

    paragraphs = soup.find_all('p')

    for p in paragraphs:
        text += p.get_text() + "\n"
        return  text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    return vector_store

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question, vector_store):
    chain = get_conversational_chain()
    
    docs = vector_store.similarity_search(user_question)
    
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    
    return response["output_text"]

def main():
    st.set_page_config("surya_guttikonda")
    st.header("Q&A with your PDF, powered by GEMINI LLM")

    user_question = st.text_input("Ask a Question, make sure the question is accurate and is in context of the uploaded PDF")

    if user_question:
        with st.spinner("Generating your embeddings..."):
            with tempfile.TemporaryDirectory() as tmpdir:
                url = st.session_state.url
                raw_text = get_url_text(url)
                text_chunks = get_text_chunks(raw_text)
                vector_store = get_vector_store(text_chunks)
                st.success("Embeddings generated.")
                answer = user_input(user_question, vector_store)
                st.write("Reply:", answer)

    with st.sidebar:
        st.title("Menu:")
        url = st.text_input("Enter Website URL:")
        
        if st.button("Submit URL"):
            st.session_state.url = url
            
            with st.spinner("Not stealing your data, just learning it....."):
                raw_text = get_url_text(url)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done!")

if __name__ == "__main__":
    main()
