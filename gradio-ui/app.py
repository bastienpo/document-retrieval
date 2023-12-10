import gradio as gr


def echo(message, history, file):
    if file is None:
        return message
    else:
        return file.name


demo = gr.ChatInterface(
    echo,
    additional_inputs=[gr.File(label="File")],
    title="Mistral 7B with RAG",
    description="Ask Yes Man any question",
    theme="soft",
)


demo.launch(server_name="0.0.0.0", server_port=7860)
