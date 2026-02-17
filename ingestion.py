import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore



load_dotenv()



if __name__ == "__main__":
    loader = PyPDFLoader("data/Rag_.pdf")
    document  = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size = 1000, chunk_overlap = 0)
    text = text_splitter.split_documents(document)

    embeddings = OllamaEmbeddings(model = "nomic-embed-text")

    PineconeVectorStore.from_documents(text, embeddings, index_name = os.environ['INDEX_NAME'])