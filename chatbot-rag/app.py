"""
Gradio UI for Mistral 7B with RAG
"""

# Author: Bastien Pouessel

from typing import List

import gradio as gr
from langchain_core.runnables.base import RunnableSequence
from logic import generate_rag_chain, load_pdf, store_vector


def initialize_chain(file: gr.File) -> RunnableSequence:
    """
    Initializes the chain with the given file.

    If no file is provided, the llm is used without RAG.

    Args:
        file (gr.File): file to initialize the chain with

    Returns:
        RunnableSequence: the chain
    """
    if file is None:
        return generate_rag_chain()

    pdf = load_pdf(file.name)
    retriever = store_vector(pdf)

    return generate_rag_chain(retriever)


def invoke_chain(message: str, history: List[str], file: gr.File = None) -> str:
    """
    Invokes the chain with the given message and updates the chain if a new file is provided.

    Args:
        message (str): message to invoke the chain with
        history (List[str]): history of messages
        file (gr.File, optional): file to update the chain with. Defaults to None.

    Returns:
        str: the response of the chain
    """
    chain = initialize_chain(file)
    return chain.invoke(message)


def create_demo() -> gr.Interface:
    """
    Creates and returns a Gradio Chat Interface.

    Returns:
        gr.Interface: the Gradio Chat Interface
    """
    return gr.ChatInterface(
        invoke_chain,
        additional_inputs=[gr.File(label="File")],
        title="Mistral 7B with RAG",
        description="Ask questions to Mistral about your pdf document.",
        theme="soft",
    )


if __name__ == "__main__":
    demo = create_demo()
    demo.queue().launch(server_name="0.0.0.0", server_port=7860)
