from langchain_community.llms import HuggingFaceHub, HuggingFaceEndpoint, Ollama
from langchain_community.chat_models.huggingface import ChatHuggingFace
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import HuggingFaceEmbeddings, HuggingFaceInferenceAPIEmbeddings, OllamaEmbeddings

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import GoogleGenerativeAI, ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_cohere import ChatCohere, CohereEmbeddings # CohereRagRetriever, CohereRerank

llms = {
    "gemini-pro": ChatGoogleGenerativeAI(model="gemini-pro", convert_system_message_to_human=True),
    "claude-3-sonnet": ChatAnthropic(model_name="claude-3-sonnet-20240229", temperature=0),
    "cohere": ChatCohere(model="command", temperature=0), # command-r-plus
    "llama2": ChatOllama(model="llama2", temperature=0),
}  

embeddings = {
    "gemini-pro": GoogleGenerativeAIEmbeddings(model="models/embedding-001"),
    "cohere": CohereEmbeddings(model="embed-english-v3.0"), # embed-multilingual-v3.0 
    "llama2": OllamaEmbeddings(model="llama2"),
}

'''
"hf_gemma-7b-it": HuggingFaceHub(repo_id="google/gemma-7b-it")
"hf_gemma-7b-it": HuggingFaceEmbeddings()
'''
