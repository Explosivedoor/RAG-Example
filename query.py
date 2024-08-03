from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaEmbeddings,OllamaLLM
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
template = """Question: {question}


Answer:"""
prompt = ChatPromptTemplate.from_template(template)


embeddings = OllamaEmbeddings(model="nomic-embed-text")
new_db = FAISS.load_local("faiss_index2", embeddings)

model = OllamaLLM(model="llama3.1")
chain = prompt | model


qa_chain = RetrievalQA.from_chain_type(llm=model, chain_type="stuff", retriever=new_db.as_retriever())

question = "How much was paid in the Namecheap invoice?"
result = qa_chain({"query": question})
print(f"Question: {question}", "\nAnswer:", result["result"])