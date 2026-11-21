
import os
import tempfile
import streamlit as st

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

st.set_page_config(
    page_title="AI PDF Study Assistant",
    page_icon="📚",
    layout="wide"
)

st.title("📚 AI PDF Study Assistant")
st.write("Upload your PDF and ask questions using AI.")

# Initialize session state
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "messages" not in st.session_state:
    st.session_state.messages = []

# Load embedding model
@st.cache_resource
def load_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

embedding_model = load_embedding_model()

# Sidebar PDF upload
with st.sidebar:
    st.header("📄 Upload your PDF")

    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"]
    )

    if st.button("Process PDF", disabled=uploaded_file is None):
        with st.spinner("Processing your PDF..."):

            # Save uploaded PDF temporarily
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".pdf"
            ) as temp_file:
                temp_file.write(uploaded_file.getvalue())
                pdf_path = temp_file.name

            try:
                # Load PDF
                loader = PyPDFLoader(pdf_path)
                docs = loader.load()

                # Split into chunks
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=100
                )
                chunks = splitter.split_documents(docs)

                # Create a separate collection for this upload
                collection_name = "pdf_" + str(
                    abs(hash(uploaded_file.name))
                )

                # Create ChromaDB vector store
                vectorstore = Chroma.from_documents(
                    documents=chunks,
                    embedding=embedding_model,
                    collection_name=collection_name,
                    persist_directory="./vectorstore"
                )

                st.session_state.vectorstore = vectorstore
                st.session_state.messages = []

                st.success("PDF processed successfully!")
                st.write(f"Pages: {len(docs)}")
                st.write(f"Chunks: {len(chunks)}")

            except Exception as e:
                st.error(f"Error processing PDF: {e}")

            finally:
                if os.path.exists(pdf_path):
                    os.remove(pdf_path)

# Chat interface
if st.session_state.vectorstore is not None:

    st.subheader("💬 Ask questions about your PDF")

    # Display previous messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_question = st.chat_input("Ask something about your PDF...")

    if user_question:

        st.session_state.messages.append({
            "role": "user",
            "content": user_question
        })

        with st.chat_message("user"):
            st.markdown(user_question)

        with st.chat_message("assistant"):

            with st.spinner("Finding your answer..."):

                retriever = st.session_state.vectorstore.as_retriever(
                    search_type="mmr",
                    search_kwargs={
                        "k": 3,
                        "fetch_k": 10,
                        "lambda_mult": 0.5
                    }
                )

                docs = retriever.invoke(user_question)

                context = "\n\n".join(
                    [doc.page_content for doc in docs]
                )

                template = ChatPromptTemplate.from_messages([
                    (
                        "system",
                        """You are an AI study assistant.
                        Answer using only the provided document context.
                        Explain in simple English.
                        If the answer is not in the context,
                        say that the uploaded PDF does not contain it."""
                    ),
                    (
                        "human",
                        "Context:\n{context}\n\nQuestion:\n{question}"
                    )
                ])

                prompt = template.invoke({
                    "context": context,
                    "question": user_question
                })

                model = ChatGroq(
                    model="openai/gpt-oss-120b",
                    temperature=0.3
                )

                response = model.invoke(prompt)
                answer = response.content

                st.markdown(answer)

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })

else:
    st.info("👈 Upload a PDF from the sidebar and click Process PDF to start.")
