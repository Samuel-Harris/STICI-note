import marimo

__generated_with = "0.8.18"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo
    from langchain_huggingface import HuggingFaceEmbeddings
    import pandas as pd
    import seaborn as sns
    import numpy as np
    return HuggingFaceEmbeddings, mo, np, pd, sns


@app.cell
def __():
    embedding_model_name = "sentence-transformers/all-mpnet-base-v2"
    test_results_dir = "embedding_model_tests"
    datasets = ["multi_passage", "single_passage", "no_answer"]
    return datasets, embedding_model_name, test_results_dir


@app.cell
def __(pd, test_results_dir):
    df = pd.read_csv(f"{test_results_dir}/test_pipelines.csv")
    df
    return (df,)


@app.cell
def __(datasets, df, pd, test_results_dir):
    results = {index: {dataset: pd.read_csv(f"{test_results_dir}/{dataset}_pipeline_{index}_responses.csv") for dataset in datasets} for index in df["index"]}
    results
    return (results,)


@app.cell
def __(HuggingFaceEmbeddings, embedding_model_name):
    embedding_model: HuggingFaceEmbeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
    return (embedding_model,)


@app.cell
def __(embedding_model, np, pd):
    def get_average_similarity_scores(dataset_response_df: pd.DataFrame):
        return np.average(np.absolute(np.array(embedding_model.embed_documents(dataset_response_df["expected_answer"])) -  np.array(embedding_model.embed_documents(dataset_response_df["actual_answer"]))))
    return (get_average_similarity_scores,)


@app.cell
def __(datasets, get_average_similarity_scores, np, results):
    average_similarity_scores = np.array([[get_average_similarity_scores(results[index][dataset]) for dataset in datasets] for index in results])
    average_similarity_scores
    return (average_similarity_scores,)


@app.cell
def __(average_similarity_scores, datasets, df, sns):
    ax = sns.heatmap(average_similarity_scores, yticklabels=df["embedding_model_name"], xticklabels=datasets)
    ax.set(xlabel="Dataset", ylabel="Embedding model")
    return (ax,)


if __name__ == "__main__":
    app.run()
