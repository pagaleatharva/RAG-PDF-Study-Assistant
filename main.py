
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

# 1. Load the embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 2. Connect to the existing ChromaDB database
vectorstore = Chroma(
    collection_name="student_study_material",
    embedding_function=embedding_model,
    persist_directory="./vectorstore"
)

print("ChromaDB connected successfully!")

# 3. Create retriever
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 3,
        "fetch_k": 10,
        "lambda_mult": 0.5
    }
)

# 4. Connect Groq LLM
model = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.3
)

# 5. Create prompt template
template = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are an AI study assistant.
        Answer the student's question using the provided context.
        Explain concepts in simple English.
        Include important concepts and key points.
        If the answer is not available in the context,
        clearly say that the document does not contain the answer."""
    ),
    (
        "human",
        """Context:
        {context}

        Question:
        {question}"""
    )
])

print("Chatbot is ready!")
print("Type 0 to exit.")

# 6. Start chatbot
while True:

    user_input = input("\nEnter your question: ")

    if user_input.strip() == "0":
        print("Exiting the chatbot.")
        break

    if not user_input.strip():
        print("Please enter a question.")
        continue

    # Retrieve relevant document chunks
    docs = retriever.invoke(user_input)

    # Combine retrieved chunks
    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    # Create final prompt
    final_prompt = template.invoke({
        "context": context,
        "question": user_input
    })

    # Generate answer
    response = model.invoke(final_prompt)

    print("\nAI:", response.content)
