from sentence_transformers import SentenceTransformer
import pinecone
import openai
import streamlit as st
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.schema import Document
from pinecone import Pinecone, ServerlessSpec
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Pinecone
from langchain_community.chat_models import ChatOpenAI
from openai import OpenAI

OPENAI_API_KEY="sk-f0hESy#$%^&CKQH"
#index_name = 'chatbot'

# Initialize SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

class PineconeConnected:
    def __init__(self, index_name: str, pinecone_api_key: str, pinecone_env: str, openai_key: str):
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)  # Use the openai_key parameter
        self.pinecone = pinecone.Pinecone(api_key='28b14#$%^&6efbe')
        self.vector_db = Pinecone.from_existing_index(index_name, embeddings)  # VectorStore object with the reference + Pinecone index loaded

    def query(self, query: str, book_title=None):  # Include self parameter
        pass

def find_match(input):
    input_em = model.encode(input).tolist()
    result = index.query(input_em, top_k=2, includeMetadata=True)
    return result['matches'][0]['metadata']['text'] + "\n" + result['matches'][1]['metadata']['text']

def query_refiner(conversation, query):

    client = OpenAI(api_key="sk-f0h#$%^&QH")
    response = client.chat.completions.create(
    model="gpt-3.5-turbo-instruct",
    prompt=f"Given the following user query and conversation log, formulate a question that would be the most relevant to provide the user with an answer from a knowledge base.\n\nCONVERSATION LOG: \n{conversation}\n\nQuery: {query}\n\nRefined Query:",
    temperature=0.7,
    max_tokens=256,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
    )
    return response['choices'][0]['text']

def get_conversation_string():
    conversation_string = ""
    for i in range(len(st.session_state['responses']) - 1):
        conversation_string += "Human: " + st.session_state['requests'][i] + "\n"
        conversation_string += "Bot: " + st.session_state['responses'][i + 1] + "\n"
    return conversation_string
