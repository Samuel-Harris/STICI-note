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
    from scipy.spatial import distance
    return HuggingFaceEmbeddings, distance, mo, np, pd, sns


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
def __(distance, embedding_model, np, pd):
    def average_cosine_similarity(dataset_response_df: pd.DataFrame):
        expected_answers_vector = np.array(embedding_model.embed_documents(dataset_response_df["expected_answer"]))
        actual_answers_vector = np.array(embedding_model.embed_documents(dataset_response_df["actual_answer"]))
        return np.average([1 - distance.cosine(expected_answers_vector[i], actual_answers_vector[i]) for i in range(len(expected_answers_vector))])
    return (average_cosine_similarity,)


@app.cell
def __(average_cosine_similarity, datasets, np, results):
    cosine_similarity_scores = np.array([[average_cosine_similarity(results[index][dataset]) for dataset in datasets] for index in results])
    cosine_similarity_scores
    return (cosine_similarity_scores,)


@app.cell
def __(cosine_similarity_scores, np):
    cosine_similarity_scores_with_average = np.column_stack((cosine_similarity_scores, np.mean(cosine_similarity_scores, axis=1)))
    cosine_similarity_scores_with_average
    return (cosine_similarity_scores_with_average,)


@app.cell
def __(cosine_similarity_scores_with_average, datasets, df, sns):
    ax = sns.heatmap(cosine_similarity_scores_with_average, yticklabels=df["embedding_model_name"], xticklabels=datasets + ["average"], cbar_kws={'label': 'Average cosine similarity'})
    ax.set(xlabel="Dataset", ylabel="Embedding model")
    return (ax,)


if __name__ == "__main__":
    app.run()
