from langchain_community.llms import HuggingFaceHub, Ollama, HuggingFaceEndpoint
from langchain_community.chat_models.huggingface import ChatHuggingFace
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import HuggingFaceEmbeddings, OllamaEmbeddings

from langchain_google_genai import GoogleGenerativeAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings

llms = {
    "gemini-pro": ChatGoogleGenerativeAI(model="gemini-pro", convert_system_message_to_human=True),
    "llama2": ChatOllama(model="llama2", temperature=0),
}  

embeddings = {
    "gemini-pro": GoogleGenerativeAIEmbeddings(model="models/embedding-001"),
    "llama2": OllamaEmbeddings(model="llama2"),
}

'''
"hf_gemma-7b-it": HuggingFaceHub(repo_id="google/gemma-7b-it")
"hf_gemma-7b-it": HuggingFaceEmbeddings()
'''
