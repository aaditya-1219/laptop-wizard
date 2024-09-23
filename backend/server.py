from transformers import pipeline
import json
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
    # Answer the question as detailed as possible from the provided context. If the question falls outside the scope of the provided context, please state that it's beyond your area of expertise."
    prompt_template = """
    If the question is related to the provided context, answer it using the context only. Otherwise give a generic answer.

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


# def user_input(user_question):
#     # Load GoogleGenerativeAIEmbeddings
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#     new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    
#     # Perform similarity search
#     docs = new_db.similarity_search(user_question)
#     chain = get_conversational_chain()  # Ensure this is defined
#     response = chain.invoke(
#         {"input_documents": docs, "question": user_question}, return_only_outputs=True)
    
#     # print(response)
#     return response["output_text"]

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    # Remove the insecure setting
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    # vector_store = FAISS.from_texts(text_chunks, embedding=embeddings,allow_dangerous_deserialization=True)  # Remove the insecure setting
    vector_store.save_local("faiss_index")

def user_input(user_question, conversation_history):
    # Load GoogleGenerativeAIEmbeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

    # Perform similarity search
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    
    # Combine conversation history and current question
    context = "\n".join(conversation_history + [user_question])
    response = chain.invoke(
        {"input_documents": docs, "question": context}, return_only_outputs=True)

    return response["output_text"]


def get_text_chunks(data):
    # Split raw_text into meaningful chunks
    # return raw_text.strip().splitlines()
    text_chunks = []
    for entry in data:
        issue = entry["issue"]
        description = entry["description"]
        potential_fixes = "\n".join(entry["potential_fixes"])
        context = entry["context"]
        chunk = f"Issue: {issue}\nDescription: {description}\nPotential Fixes:\n{potential_fixes}\nContext: {context}\n"
        text_chunks.append(chunk)
    return text_chunks

def main():
    conversation_history = []
    # # Your student data as raw text
    # raw_text = """
    # STUDENT_DATA = {
    #     "Rohan": {"age": 15, "score": 85},
    #     "Anjali": {"age": 14, "score": 90},
    #     "Kailash": {"age": 16, "score": 78},
    #     "Arpit": {"age": 14, "score": 92},
    #     "Vivek": {"age": 15, "score": 88},
    # }
    # """

    # # Process to create vector store
    # text_chunks = get_text_chunks(raw_text)
    # print("Length of text_chunks:", len(text_chunks))
    # print("Contents of text_chunks:", text_chunks)
    
    # get_vector_store(text_chunks)

    # # Example user question
    # # user_question = "What is the age of Rohan?" # works
    # user_question = "What can I do if my laptop won't turn on?"
    # response = user_input(user_question)
    # print("Response:", response)
    # Load your JSON data from a file or directly in your code
    raw_data = """
    [
        {
            "issue": "Laptop won't turn on",
            "description": "User reports that the laptop does not power up, no lights are on.",
            "potential_fixes": ["Check if the power adapter is connected properly.", "Try a different power outlet.", "Remove the battery and hold the power button for 15 seconds, then reinsert the battery."], 
            "context": "Common issue that can be caused by power supply problems or hardware failures."
        },
        {
            "issue": "Overheating", "description": "Laptop becomes very hot and the fan runs loudly.", 
            "potential_fixes": ["Ensure vents are not blocked and clean dust from the fan.", "Use a cooling pad.", "Check for resource-heavy applications running in the background."],
            "context": "Overheating can lead to performance issues and hardware damage."
        }
    ]
    """
    
    data = json.loads(raw_data)
    text_chunks = get_text_chunks(data)

    # Process to create vector store
    get_vector_store(text_chunks)

    # Example user question
    # user_question = "What can I do if my laptop won't turn on?"
    # response = user_input(user_question)
    # print("Response:", response)
    while True:
        user_question = input("You: ")
        conversation_history.append(f"You: {user_question}")  # Store user's question

        response = user_input(user_question, conversation_history)
        conversation_history.append(f"AI: {response}")  # Store model's response
        
        print("AI:", response)

if __name__ == "__main__":
    main()
