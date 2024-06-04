# STICI-note
Semantic and Textual Inference Chatbot Interface (STICI)-note is a chatbot interface for querying textual data designed to be completely locally runnable on an M1 macbook.

Read about it in my blog: [Semantic and Textual Inference Chatbot Interface (STICI-Note) - Part 1: Planning and Prototyping](https://bytes-and-nibbles.web.app/bytes/stici-note-part-1-planning-and-prototyping).

## Prototype
The prototype/ directory contains the Jupyter notebook used to develop the proof-of-concept of STICI-note (**prototype.ipynb**) and the library versions that I used (**environment.yml**).

The prototype is a simple RAG application that tests querying chunks of Wikipedia pages stored in an in-memory vector DB.

To run:
1. Download the Q6_K quantised TinyLlama .gguf file from https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/blob/main/tinyllama-1.1b-chat-v1.0.Q6_K.gguf.
2. Install the appropriate libraries using `conda env update -f environment.yml` or your preferred Python package manager.
3. Run the notebook using `jupyter lab`.