import os

from dotenv import load_dotenv
from operator import itemgetter
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

embeddings = OllamaEmbeddings(model = "nomic-embed-text")

llm = ChatOllama(model = "gpt-oss:20b", temperature = 0)

vectorsstore = PineconeVectorStore(embedding = embeddings,index_name=os.environ['INDEX_NAME'])

retriever = vectorsstore.as_retriever(search_kwargs={"k": 3})

prompt_template = ChatPromptTemplate.from_template(
    """Answer the question based only on the following context:
    {context}
    Question: {question}
    
    Provide  a detailed answer:"""
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def create_retrival_chain():
    retrieval_chain = (
        RunnablePassthrough.assign(
            context = itemgetter("question") | retriever | format_docs
        )
        | prompt_template
        | llm
        | StrOutputParser()
    )
    return retrieval_chain

if __name__ == "__main__":
    print("retrieving...")

    query = "What is RAG and how does it work?"

    chain = create_retrival_chain()

    chain_result = chain.invoke({"question": query})
    print(chain_result)