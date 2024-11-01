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
    test_results_dir = "llm_tests"
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
    for results_df in results.values():
        for dataset in datasets:
            results_df[dataset]["actual_answer"] = results_df[dataset]["actual_answer"].astype(str)
    results
    return dataset, results, results_df


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
def __(average_similarity_scores, np):
    len(np.mean(average_similarity_scores, axis=1))
    return


@app.cell
def __(average_similarity_scores):
    len(average_similarity_scores)
    return


@app.cell
def __(average_similarity_scores, np):
    new_average_similarity_scores = np.column_stack((average_similarity_scores, np.mean(average_similarity_scores, axis=1)))
    new_average_similarity_scores
    return (new_average_similarity_scores,)


@app.cell
def __(datasets, df, new_average_similarity_scores, sns):
    ax = sns.heatmap(new_average_similarity_scores, yticklabels=df["model_path"], xticklabels=datasets + ["average"])
    ax.set(xlabel="Dataset", ylabel="LLM")
    return (ax,)


if __name__ == "__main__":
    app.run()
