"""
Gradio UI for Mistral 7B with RAG
"""

# Author: Bastien Pouessel

import logging

import gradio as gr
import requests

logging.basicConfig(level=logging.DEBUG)


def update_file(file):
    # change file for
    if file is not None:
        requests.post("http://localhost:80/document", json={"path": file.name})

        logging.debug(f"File was uploaded: {file.name}")


def echo(message, history, file):
    # change file for
    update_file(file)

    # call the chain through the API
    payload = {"input": message}
    response = requests.post(
        "http://localhost:80/chain/invoke",
        json=payload,
    )

    logging.debug(f"Response: {response.json()}")

    if "output" in response.json():
        return response.json()["output"]

    return None


demo = gr.ChatInterface(
    echo,
    additional_inputs=[gr.File(label="File")],
    title="Mistral 7B with RAG",
    description="Ask Yes Man any question",
    theme="soft",
)


demo.queue().launch(server_name="0.0.0.0", server_port=7860)
