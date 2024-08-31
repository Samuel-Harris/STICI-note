import marimo

__generated_with = "0.7.9"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo
    import dspy
    from dspy import Example
    from dspy.evaluate import Evaluate
    from dspy.datasets import gsm8k
    from dspy.teleprompt import BootstrapFewShotWithRandomSearch
    from dspy.retrieve.chromadb_rm import ChromadbRM
    from langchain_huggingface import HuggingFaceEmbeddings
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
    import pandas as pd
    return (
        ChromadbRM,
        Example,
        HuggingFaceEmbeddings,
        SentenceTransformerEmbeddingFunction,
        dspy,
        mo,
        pd,
    )


@app.cell
def __():
    import os
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    os.environ["ACCELERATE_USE_MPS_DEVICE"] = "True"
    return os,


@app.cell
def __():
    import torch
    torch.backends.mps.is_available()
    torch.set_default_device("mps")
    torch.get_default_device()
    return torch,


@app.cell
def __(dspy):
    tiny_llama = dspy.HFModel(model="microsoft/Phi-3-mini-4k-instruct")
    return tiny_llama,


@app.cell
def __(
    ChromadbRM,
    HuggingFaceEmbeddings,
    SentenceTransformerEmbeddingFunction,
):
    embeddings_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    retriever_model = ChromadbRM(
        "example_collection",
        'example_dbb',
        embedding_function=SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2", device="mps"),
        k=5
    )
    return embeddings_function, retriever_model


@app.cell
def __(dspy, retriever_model, tiny_llama):
    dspy.settings.configure(lm=tiny_llama, rm=retriever_model)
    return


@app.cell
def __(pd):
    documents_df = pd.read_csv("../data/documents.csv")
    no_answer_questions_df = pd.read_csv("../data/no_answer_validate_questions.csv")
    single_passage_questions_df = pd.read_csv("../data/single_passage_validate_questions.csv")
    multi_passage_questions_df = pd.read_csv("../data/multi_passage_validate_questions.csv")
    return (
        documents_df,
        multi_passage_questions_df,
        no_answer_questions_df,
        single_passage_questions_df,
    )


@app.cell
def __(
    Example,
    documents_df,
    multi_passage_questions_df,
    no_answer_questions_df,
    single_passage_questions_df,
):
    no_answer_response = "The answer to your question is not in the provided document."
    no_answer_questions = [Example(question=row[1]["question"],
                                   answer=no_answer_response,
                                   document=documents_df[documents_df["index"] == row[1]["document_index"]]["text"].iloc[
                                       0]).with_inputs("question", "document")
                           for row in no_answer_questions_df.iterrows()]
    single_passage_questions = [Example(question=row[1]["question"],
                                        answer=row[1]["answer"],
                                        document=
                                        documents_df[documents_df["index"] == row[1]["document_index"]]["text"].iloc[
                                            0]).with_inputs("question", "document")
                                for row in single_passage_questions_df.iterrows()]
    multi_passage_questions = [Example(question=row[1]["question"],
                                       answer=row[1]["answer"],
                                       document=
                                       documents_df[documents_df["index"] == row[1]["document_index"]]["text"].iloc[
                                           0]).with_inputs("question", "document")
                               for row in multi_passage_questions_df.iterrows()]
    validate_questions = no_answer_questions + single_passage_questions + multi_passage_questions
    return (
        multi_passage_questions,
        no_answer_questions,
        no_answer_response,
        single_passage_questions,
        validate_questions,
    )


@app.cell
def __(validate_questions):
    validate_questions[0].to("mps")
    return


@app.cell
def __(dspy):
    class GenerateAnswer(dspy.Signature):
        """Answer questions with short factoid answers."""

        context = dspy.InputField(desc="may contain relevant facts")
        question = dspy.InputField()
        answer = dspy.OutputField(desc="if there is an answer, it will consist of information from the context")
    return GenerateAnswer,


@app.cell
def __(GenerateAnswer, dspy):
    class RAG(dspy.Module):
        def __init__(self, num_passages=3):
            super().__init__()

            self.retrieve = dspy.Retrieve(k=num_passages)
            self.generate_answer = dspy.ChainOfThought(GenerateAnswer)

        def forward(self, question, document):
            context = self.retrieve(question).passages
            prediction = self.generate_answer(context=context, question=question)
            return dspy.Prediction(context=context, answer=prediction.answer)
    return RAG,


@app.cell
def __():
    from dspy.teleprompt import BootstrapFewShot
    return BootstrapFewShot,


@app.cell
def __():
    return


@app.cell
def __(dspy):
    # Validation logic: check that the predicted answer is correct.
    # Also check that the retrieved context does actually contain that answer.
    def validate_context_and_answer(example, pred, trace=None):
        answer_EM = dspy.evaluate.answer_exact_match(example, pred)
        answer_PM = dspy.evaluate.answer_passage_match(example, pred)
        return answer_EM and answer_PM
    return validate_context_and_answer,


@app.cell
def __(BootstrapFewShot, validate_context_and_answer):
    # Set up a basic teleprompter, which will compile our RAG program.
    teleprompter = BootstrapFewShot(metric=validate_context_and_answer)
    return teleprompter,


@app.cell
def _(RAG, teleprompter, validate_questions):
    # Compile!
    compiled_rag = teleprompter.compile(RAG(), trainset=validate_questions)
    return compiled_rag,


if __name__ == "__main__":
    app.run()
