from fastapi import FastAPI
from langserve import add_routes
from llm_retrival import load_pdf, store_vector, generate_rag_chain


# FastAPI app
app = FastAPI(
  title="Document Chatbot",
  version="0.1",
  description="API server in order to retrieve information from a document",
)

# Load the document and generate the chain
document_path = "data/thesis.pdf"
all_splits = load_pdf(document_path)
retriever = store_vector(all_splits)
chain = generate_rag_chain(retriever)

# Add routes to the app using the chain
add_routes(
    app,
    chain,
    path="/chain",
)