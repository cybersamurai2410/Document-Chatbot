from langchain_community.llms import HuggingFaceHub, HuggingFaceEndpoint, Ollama
from langchain_community.chat_models.huggingface import ChatHuggingFace
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import HuggingFaceEmbeddings, HuggingFaceInferenceAPIEmbeddings, OllamaEmbeddings

from langchain_openai import ChatOpenAI, OpenAIEmbeddings # Also used for TogetherAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_cohere import ChatCohere, CohereEmbeddings # CohereRagRetriever, CohereRerank
from langchain_groq import ChatGroq
# from langchain_mistralai import ChatMistralAI
# from langchain_google_vertexai import ChatVertexAI
# from langchain_fireworks import ChatFireworks

llms = {
    "gpt-4o-mini (openai)": ChatOpenAI(model_name='gpt-4o-mini'),
    "gemini-1.0-pro (google)": ChatGoogleGenerativeAI(model="gemini-1.0-pro", convert_system_message_to_human=True),
    "claude-3.5-haiku (anthropic)": ChatAnthropic(model_name="claude-3-5-haiku-20241022"), # claude-3-5-sonnet-20241022
    "cohere-command (cohere)": ChatCohere(model="command"), 
    "groq-llama-3-8b (meta)": ChatGroq(model_name="llama3-8b-8192"),
}  

embeddings = {
    "openai_gpt": OpenAIEmbeddings(model="text-embedding-3-small"),
    "google-gemini": GoogleGenerativeAIEmbeddings(model="models/embedding-001"),
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
