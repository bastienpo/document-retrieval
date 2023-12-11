"""
Gradio UI for Mistral 7B with RAG
"""

# Author: Bastien Pouessel

import gradio as gr

def echo(message, history, file):
    print(message)
    return message


demo = gr.ChatInterface(
    echo,
    additional_inputs=[gr.File(label="File")],
    title="Mistral 7B with RAG",
    description="Ask questions to Mistral about your pdf document.",
    theme="soft",
)


demo.queue().launch(server_name="0.0.0.0", server_port=7860)
