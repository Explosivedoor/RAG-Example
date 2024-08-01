from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings,OllamaLLM


model = OllamaLLM(model="llama3.1")

embeddings = OllamaEmbeddings(model="nomic-embed-text")

DATA_PATH = "\data"

loader= DirectoryLoader(DATA_PATH)
documents = loader.load()


text_splitter = RecursiveCharacterTextSplitter(
chunk_size=300,
chunk_overlap=100,
length_function=len,
add_start_index=True,
)

chunks = text_splitter.split_documents(documents)
print(f"Split {len(documents)} documents into {len(chunks)} chunks.")


db = FAISS.from_documents(chunks, embedding=embeddings)
db.save_local("faiss_index")


