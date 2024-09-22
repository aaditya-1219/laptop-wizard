from transformers import pipeline
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_conversational_chain():
    # prompt_template = """
    # Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    # provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    # Context:\n {context}?\n
    # Question: \n{question}\n

    # Answer:
    # """
    prompt_template = """
    You are a knowledgeable assistant. Answer the question as detailed as possible from the provided context. If the question falls outside the scope of the provided context, please state that it's beyond your area of expertise."

    Context:
    {context}
    
    Question: 
    {question}

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template,
                            input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain


def user_input(user_question):
    # Load GoogleGenerativeAIEmbeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    
    # Perform similarity search
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()  # Ensure this is defined
    response = chain.invoke(
        {"input_documents": docs, "question": user_question}, return_only_outputs=True)
    
    # print(response)
    return response["output_text"]

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    # Remove the insecure setting
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    # vector_store = FAISS.from_texts(text_chunks, embedding=embeddings,allow_dangerous_deserialization=True)  # Remove the insecure setting
    vector_store.save_local("faiss_index")


def get_text_chunks(raw_text):
    # Split raw_text into meaningful chunks
    return raw_text.strip().splitlines()

def main():
    # Your student data as raw text
    raw_text = """
    STUDENT_DATA = {
        "Rohan": {"age": 15, "score": 85},
        "Anjali": {"age": 14, "score": 90},
        "Kailash": {"age": 16, "score": 78},
        "Arpit": {"age": 14, "score": 92},
        "Vivek": {"age": 15, "score": 88},
    }
    """
    
    # Process to create vector store
    text_chunks = get_text_chunks(raw_text)
    print("Length of text_chunks:", len(text_chunks))
    print("Contents of text_chunks:", text_chunks)
    
    get_vector_store(text_chunks)

    # Example user question
    # user_question = "What is the age of Rohan?" # works
    user_question = "what is the capital of France?"
    response = user_input(user_question)
    print("Response:", response)

if __name__ == "__main__":
    main()
