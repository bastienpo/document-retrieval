"""
This script is used to generate a RAG chain to answer questions about a pdf document.
"""

# Author: Bastien Pouessel

import os

from langchain.document_loaders import UnstructuredPDFLoader
from langchain.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain.llms import HuggingFaceEndpoint
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

HF_API_KEY = os.environ["HF_API_KEY"]


class MistralOutputParser(StrOutputParser):
    """OutputParser that parser llm result from Mistral API"""

    def parse(self, text: str) -> str:
        """
        Returns the input text with no changes.

        Args:
            text (str): text to parse

        Returns:
            str: parsed text
        """
        return text.split("[/INST]")[-1].strip()


def load_pdf(
    document_path: str,
    mode: str = "single",
    strategy: str = "fast",
    chunk_size: int = 500,
    chunk_overlap: int = 0,
):
    """
    Load a pdf document and split it into chunks of text.

    Args:
        document_path (Path): path to the pdf document
        mode (str, optional): mode of the loader. Defaults to "single".
        strategy (str, optional): strategy of the loader. Defaults to "fast".
        chunk_size (int, optional): size of the chunks. Defaults to 500.
        chunk_overlap (int, optional): overlap of the chunks. Defaults to 0.

    """

    # Load the document
    loader = UnstructuredPDFLoader(
        document_path,
        mode=mode,
        strategy=strategy,
    )

    docs = loader.load()

    # Split the document into chunks of text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    all_splits = text_splitter.split_documents(docs)

    return all_splits


def store_vector(all_splits):
    """
    Store vector of each chunk of text.

    Args:
        all_splits ([type]): All splits of the document
    """

    # Use the HuggingFace distilbert-base-uncased model to embed the text
    embeddings_model_url = (
        "https://api-inference.huggingface.co/models/distilbert-base-uncased"
    )

    embeddings = HuggingFaceInferenceAPIEmbeddings(
        endpoint_url=embeddings_model_url,
        api_key=HF_API_KEY,
    )

    # Store the embeddings of each chunk of text into ChromaDB
    vector_store = Chroma.from_documents(all_splits, embeddings)
    retriever = vector_store.as_retriever()

    return retriever


def generate_mistral_prompt() -> ChatPromptTemplate:
    """
    Generate a prompt for Mistral API.

    Returns:
        ChatPromptTemplate: prompt for Mistral API
    """
    template = "[INST] {context} {prompt} [/INST]"
    prompt_template = ChatPromptTemplate.from_template(template)
    return prompt_template


def generate_rag_chain(retriever):
    """
    Generate a RAG chain.

    Args:
        retriver ([type]): retriever

    Returns:
        [type]: RAG chain
    """

    # Retrive document from chromaDB
    retrival = {"context": retriever, "prompt": RunnablePassthrough()}

    prompt_template = generate_mistral_prompt()

    # Use the Mistral Free prototype API
    mistral_url = (
        "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1"
    )

    model_endpoint = HuggingFaceEndpoint(
        endpoint_url=mistral_url,
        huggingfacehub_api_token=HF_API_KEY,
        task="text2text-generation",
    )

    output_parser = MistralOutputParser()

    # Define the chain of components
    chain = retrival | prompt_template | model_endpoint | output_parser

    return chain


if __name__ == "__main__":
    document_path = "data/thesis.pdf"
    all_splits = load_pdf(document_path)
    retriever = store_vector(all_splits)
    chain = generate_rag_chain(retriever)

    response = chain.invoke("What are the names of the supervisors of this thesis?")

    print(response)
