import torch
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings

from evaluation.evaluation_suite.model_config import ModelConfig
from evaluation.vector_db.vector_db import construct_chroma_client
from evaluation.pipelines.basic_pipeline import BasicPipeline
from evaluation.models.model_factory import construct_hf_model

raw_prompt_template = """<|system|>
In this conversation between a user and the AI, the AI is helpful and friendly, and when it does not know the answer it says "I don’t know".

To help answer the question, you can use the following information:
{context}</s>
<|user|>
{input}</s>
<|AI|>
"""
prompt_template = PromptTemplate(template=raw_prompt_template, input_variables=["input", "context"])

print(f"Is torch available on MPS: {torch.backends.mps.is_available()}")

test_config = ModelConfig(model_path="models/model_weights/tinyllama-1.1b-chat-v1.0.Q6_K.gguf",
                          temperature=0.75,
                          max_tokens=2000,
                          top_p=1,
                          top_k=40,
                          last_n_tokens_size=64,
                          n_ctx=2048,
                          n_batch=512,
                          n_gpu_layers=-1,
                          f16_kv=True)

model = construct_hf_model(test_config)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
text_chunks = [Document("My favourite big cat is the lion."), Document("Koalas are my favourite animal")]
vector_db = construct_chroma_client(text_chunks, embeddings)

basic_pipeline = BasicPipeline(model, prompt_template, vector_db.as_retriever())

print(basic_pipeline.query("What is my favourite animal?"))
