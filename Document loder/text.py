
from langchain_community.document_loaders import PyPDFLoader, TextLoader

# Load PDF document
data = PyPDFLoader("Document loder/FINAL_forth_year 2.pdf")

# Extract document pages
docs = data.load()

# Print first page content
print(docs[1].page_content)
