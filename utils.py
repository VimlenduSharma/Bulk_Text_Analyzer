import re
import xmltodict
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI

# Ensure_NLTK_data_is_downloaded
nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    """
    Preprocess the input text by removing numbers, short words, punctuation,
    and stopwords.
    """
    text = re.sub(r'\d+', '', text)  
    text = re.sub(r'\b\w{1,2}\b', '', text) 
    text = re.sub(r'[^\w\s]', '', text) 
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token.lower() for token in tokens if token.lower() not in stop_words]
    return ' '.join(filtered_tokens)

def split_text_into_chunks(text, chunk_size=1000, chunk_overlap=200):
    """
    Split the text into smaller chunks for efficient processing.
    """
    text_splitter = CharacterTextSplitter(separator=" ", chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_text(text)

#vector_store_creation
def create_vector_store_from_text_chunks(text_chunks, openai_api_key):
    """
    Create a vector store from text chunks using OpenAI embeddings and FAISS.
    """
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vector_store = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vector_store

#conversational_retrieval_chain
def get_conversation_chain(vector_store, openai_api_key):
    """
    Initialize the conversational retrieval chain using LangChain and OpenAI.
    """
    llm = ChatOpenAI(api_key=openai_api_key, model="gpt-4-turbo")
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever()
    )
    return conversation_chain
