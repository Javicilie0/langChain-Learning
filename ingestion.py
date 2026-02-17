import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader, PyPDFLoader,Docx2txtLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore



load_dotenv()

if __name__ == "__main__":
    pdfLoader = PyPDFLoader("data/Rag_.pdf")
    docLoader = Docx2txtLoader("data/LLM.docx")
    txtLoader = TextLoader("data/MachineLearning.txt")

    documents = pdfLoader.load() + docLoader.load() + txtLoader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    text = text_splitter.split_documents(documents)

    embeddings = OllamaEmbeddings(model= "nomic-embed-text")

    PineconeVectorStore.from_documents(text, embeddings, index_name=os.getenv("INDEX_NAME"))
