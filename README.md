# Document retrival chabot

The following project is a proof of concept for a retrieval-augmented chatbot. In this case, it's utilizing the Mistral 7b Instruct model with the Hugging Face prototyping API and includes a Gradio UI for interacting with the model. The pipeline is based on LangChain and supports PDF as a document type. The advantage of retrieval-augmented generation is that it helps reduce model hallucinations.

I chose LangChain for its simplicity in integrating all the components together and opted to use only models through the API due to computational requirements. DistilBERT was used as the embedding model, while Mistral was used for the language generation part.

The RAG methodology requires in-memory storage to store document embeddings that have been divided into multiple parts. I selected ChromaDB for its ease of use and the capability to integrate it seamlessly.

## Project architecture

The project can be deployed with docker compose and is only composed of parts that are dockerized. The following architecture provide on overview of the project.

<p align="center">
  <img src="docs/architecture.png" width="950" height="275">
</p>

The project is composed of the following parts:
- gradio-ui: This contains the project's user interface and is deployed using Docker.
- langserve: This is the core component that houses the logic for RAG and storage. It is stored in the 'chatbot-rag' directory and uses an API provided by LangServe to expose a server for your chain.

### How to run the project
The project can be easily run with Docker Compose, but in order to function, a `.env` file must be present alongside the `docker-compose.yml` file with a key for Hugging Face. The format of the environment variable is as follows:"
```bash
HF_API_KEY=hf_<rest of the key>
```

The following command will execute the docker-compose.yml
```
docker compose up -d
```

The UI will be available at the following address: `http://localhost:7860/`
