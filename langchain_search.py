import os
from pathlib import Path
from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS

# Replace with your own API key
os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY"

# Load all Markdown files from the 'conversaciones' folder
documents = []
for file_path in Path("conversaciones").glob("*.md"):
    loader = TextLoader(str(file_path), encoding="utf-8")
    documents.extend(loader.load())

# Split documents into smaller chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = splitter.split_documents(documents)

# Create the vector store with embeddings
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(docs, embeddings)

def search(query, k=3):
    """Return the k most relevant fragments for the given query."""
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    results = retriever.get_relevant_documents(query)
    return [doc.page_content for doc in results]

if __name__ == "__main__":
    question = input("Write your question: ")
    for fragment in search(question):
        print("=" * 40)
        print(fragment)
