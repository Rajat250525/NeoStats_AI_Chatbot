from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def create_rag_retriever(file_path, embedding_model):
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    docs = splitter.split_documents(pages)
    vectordb = FAISS.from_documents(docs, embedding_model)
    return vectordb.as_retriever()

def get_rag_tool(retriever):
    def rag_tool(query):
        docs = retriever.get_relevant_documents(query)
        return "\n".join([d.page_content[:400] for d in docs])
    return rag_tool
