from langchain_community.llms import HuggingFaceHub, HuggingFaceEndpoint, Ollama
from langchain_community.chat_models.huggingface import ChatHuggingFace
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import HuggingFaceEmbeddings, HuggingFaceInferenceAPIEmbeddings, OllamaEmbeddings

from langchain_openai import ChatOpenAI # Also used for TogetherAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import GoogleGenerativeAI, ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_cohere import ChatCohere, CohereEmbeddings # CohereRagRetriever, CohereRerank
from langchain_groq import ChatGroq
# from langchain_mistralai import ChatMistralAI
# from langchain_google_vertexai import ChatVertexAI
# from langchain_fireworks import ChatFireworks

llms = {
    "gemini-pro": ChatGoogleGenerativeAI(model="gemini-pro", convert_system_message_to_human=True),
    "claude-3-sonnet": ChatAnthropic(model_name="claude-3-sonnet-20240229", temperature=0),
    "cohere": ChatCohere(model="command", temperature=0), # command-r-plus
    "groq-llama3_8b": ChatGroq(temperature=0, model_name="llama3-8b-8192"),
    "groq-gemma7b": ChatGroq(temperature=0, model_name="gemma-7b-it"),
    "gpt-3.5-turbo": 'ChatOpenAI(temperature=0)'
}  

embeddings = {
    "gemini-pro": GoogleGenerativeAIEmbeddings(model="models/embedding-001"),
    "cohere": CohereEmbeddings(model="embed-english-v3.0"), # embed-multilingual-v3.0 
}

'''
local models:
"llama2": ChatOllama(model="llama2", temperature=0),
"llama2": OllamaEmbeddings(model="llama2"),

huggingface:
"hf_gemma-7b-it": HuggingFaceHub(repo_id="google/gemma-7b-it"),
"hf_gemma-7b-it": HuggingFaceEmbeddings(),
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
'''
