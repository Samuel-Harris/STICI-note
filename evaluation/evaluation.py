from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings

from vector_db.vector_db import construct_chroma_client
from pipelines.basic_pipeline import BasicPipeline
from models.model_loaders import load_llama_cpp_model

raw_prompt_template = """<|system|>
In this conversation between a user and the AI, the AI is helpful and friendly, and when it does not know the answer it says "I don’t know".

To help answer the question, you can use the following information:
{context}</s>
<|user|>
{input}</s>
<|AI|>33
"""
prompt_template = PromptTemplate(template=raw_prompt_template, input_variables=["input", "context"])

model = load_llama_cpp_model("../prototype/tinyllama-1.1b-chat-v1.0.Q6_K.gguf", 0.75, 1)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
text_chunks = [Document("My favourite big cat is the lion."), Document("Koalas are my favourite animal")]
vector_db = construct_chroma_client(text_chunks, embeddings)

basic_pipeline = BasicPipeline(model, prompt_template, vector_db.as_retriever())

print(basic_pipeline.query("What is my favourite animal?"))
