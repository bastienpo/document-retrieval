from fastapi import FastAPI, Response
from langserve import add_routes
from llm_retrival import generate_rag_chain, load_pdf, store_vector
from pydantic import BaseModel


class Document(BaseModel):
    """
    Document path model.
    """

    path: str


# FastAPI app
app = FastAPI(
    title="Document Chatbot",
    version="0.1",
    description="API server in order to retrieve information from a document",
)

app.all_splits = load_pdf("data/thesis.pdf")
app.retriever = store_vector(app.all_splits)
app.chain = generate_rag_chain(app.retriever)


@app.post("/document")
def set_document(document: Document) -> Response:
    """
    Set the document to use for the chatbot.

    Args:
        document_path (str): path to the document
    """

    # Split the document into chunks of text set a new retriever and chain
    app.all_splits = load_pdf(document.path)
    app.retriever = store_vector(app.all_splits)
    app.chain = generate_rag_chain(app.retriever)
    return {"message": "Document set"}


# Add routes to the app using the chain
add_routes(
    app,
    app.chain,
    path="/chain",
)
