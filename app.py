import streamlit as st
from PyPDF2 import PdfReader

import os
from dotenv import load_dotenv
import google.generativeai as genai


# used in get_chunks
from langchain.text_splitter import RecursiveCharacterTextSplitter

# used in  vector_embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS

# used in conversation_chain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain

load_dotenv()

os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def extract_text_pdf(pdfs):   #extract text from each pdf and store it in dictionary with key : pdf name , value : content
    text_dict = {}
    for pdf in pdfs: 
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        text_dict[pdf.name] = text
    return text_dict

def get_chunks(text): # split the text into chunk size of 5000 with overlap allowed 1000

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=1000) #initiaizing an object of RecursiveCharacterTextSplitter class

    chunks = text_splitter.split_text(text)
    
    return chunks

def vector_embeddings(chunks):
    
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/text-embedding-004")

    # Facebook AI Similarity Search
    local_vector_store = FAISS.from_texts(chunks, embedding=embeddings)   #local vector that can be efficient to search on your text chunks , it uses indexing
    
    local_vector_store.save_local("faiss_index") # saved as faiss_index


def conversation_chain():

    prompt_template = """

    Pls give me detailed and related answer for question from from the provided context.

    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:

    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3) # here temperature is randomness

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

def ask_question(user_question):

    embeddings = GoogleGenerativeAIEmbeddings(model = "models/text-embedding-004")

    try:
        context = FAISS.load_local("faiss_index", embeddings,allow_dangerous_deserialization=True)
    except ValueError as e:
        print(f"Error loading FAISS index: {e}")

    context = FAISS.load_local("faiss_index", embeddings,allow_dangerous_deserialization=True)    
    # Deserialization refers to the process of converting a data format
    # (e.g., JSON, pickle, etc.) back into an object in memory. T

    if context:
        
        best_context = context.similarity_search(user_question)
        chain = conversation_chain()  # initiaize an object and this chain contains loading of model to prompt
        
        with st.spinner('Loading...'):
            response = chain(
            {"input_documents":best_context, "question": user_question}
            , return_only_outputs=True)
        
        st.write("Response ", response["output_text"])

        st.subheader("Portion of pdf relevant to answer")

        # best_context is list of object : [Document(id , meta , content)] 

        content_display = best_context[0].page_content
        to_be_highlighted = response["output_text"]
        print(to_be_highlighted)
        
        highlighted_content = content_display.replace(
            to_be_highlighted,
            f'<span style="background-color: red; color: white;">{to_be_highlighted}</span>') 
        
        st.markdown(highlighted_content,unsafe_allow_html=True)
        # searched form net 
        # Setting unsafe_allow_html=True tells Streamlit to allow H
        # TML content within the Markdown.


def main():

    st.set_page_config("Chat PDF")

    pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True) # Allow users to upload a PDF document.
    
    if pdf_docs:
        
        extracted_texts = extract_text_pdf(pdf_docs)     #Extract text from the PDF (use libraries like PyPDF2 , pdfplumber or Langchain).
        for file_name, text in extracted_texts.items():
            with st.expander(f"Click to view content from {file_name}"):
                st.text_area(f"{file_name} Text", text, height=300)


        if st.button("Submit & Process"):
            with st.spinner("Processing..."):

                # Convert the text to embeddings (e.g., using Gemini,sentence-transformers or OpenAI embeddings).
                all_chunks = []
                for text in extracted_texts.values():
                    chunks = get_chunks(text)
                    all_chunks.extend(chunks)

                vector_embeddings(all_chunks)
                st.success("Done")

    st.subheader("Chat with PDF's")
    user_question = st.text_input("Ask Questions")

    if user_question and pdf_docs:
        # Enable users to ask questions about the document's content, with responses generated via an LLM (e.g., Gemini , GPT-3.5 or Hugging Face models).
        ask_question(user_question)


if __name__ == "__main__":
    main()



# Flow 
# let's consider i upload a pdf which contains 100k words
# then it will split the text into 10 parts i.e called chunks
# with size of 10k , then each chunk will futher convert into embeddings using model
# when we input a qurstion , it is also convert into embedding.
# embedding is a vector to get semantic meaning 
# now we have 1 question vector , 10 context vectors (chunk )
# we will choose 1 bext related context vector and 1 question vector to pass as prompt 
# to model , now using similarity (to find similiarity of two vectors we use dot product)
# by passing 1 context vector and 1 question vector to model we get response

# initial dev setup by me on MAC with commands

# create a dir : STRING_VENTURES_ASSGN
# create a virtual env : python -m venv venv
# create requirements.txt
# install all req : pip install -r requirements.txt
# write code in app.py