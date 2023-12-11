"""
Gradio UI for Mistral 7B with RAG
"""

# Author: Bastien Pouessel

import gradio as gr
from llm_retrival import load_pdf, generate_rag_chain, store_vector

def initialize_chain(file):
    """
    Initializes the chain with the given file.
    """

    if file is None:
        return generate_rag_chain()
    
    pdf = load_pdf(file.name)
    retriever = store_vector(pdf) 

    return generate_rag_chain(retriever)

def invoke_chain(message, history, file=None) -> str:
    """
    Invokes the chain with the given message and updates the chain if a new file is provided.
    """
    chain = initialize_chain(file)
    return chain.invoke(message)

def create_demo() -> gr.Interface:
    """
    Creates and returns a Gradio Chat Interface.
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
