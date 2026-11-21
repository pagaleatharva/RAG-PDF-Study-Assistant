
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# 1. Load PDF
loader = PyPDFLoader("Document loder/FINAL_forth_year 2.pdf")
docs = loader.load()

print("PDF loaded successfully!")
print("Total pages:", len(docs))

# 2. Split documents into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)

chunks = splitter.split_documents(docs)
print("Total chunks:", len(chunks))

# 3. Load embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Embedding model loaded successfully!")

# 4. Store documents in ChromaDB
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory="./vectorstore",
    collection_name="student_study_material"
)

print("Documents stored successfully in ChromaDB!")

# 5. Test similarity search
results = vectorstore.similarity_search(
    "Explain the important concepts in this document.",
    k=3
)

for i, result in enumerate(results, 1):
    print(f"\n--- Result {i} ---")
    print(result.page_content)
