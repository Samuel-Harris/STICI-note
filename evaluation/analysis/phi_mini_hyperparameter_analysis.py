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
    test_results_dir = "hyperparameter_tests"
    datasets = ["multi_passage", "single_passage", "no_answer"]
    return datasets, embedding_model_name, test_results_dir


@app.cell
def __(pd, test_results_dir):
    df_1 = pd.read_csv(f"{test_results_dir}/run_1/test_pipelines.csv")
    df_1
    return (df_1,)


@app.cell
def __(pd, test_results_dir):
    df_2 = pd.read_csv(f"{test_results_dir}/run_2/test_pipelines.csv")
    df_2
    return (df_2,)


@app.cell
def __(df_1, df_2):
    df_2["index"] += max(df_1["index"]) + 1
    df_2
    return


@app.cell
def __(df_1, df_2, pd):
    df = pd.concat([df_1, df_2])
    df
    return (df,)


@app.cell
def __(datasets, df_1, pd, test_results_dir):
    results_1 = {index: {dataset: pd.read_csv(f"{test_results_dir}/run_1/{dataset}_pipeline_{index}_responses.csv") for dataset in datasets} for index in df_1["index"]}
    for results_df in results_1.values():
        for dataset in datasets:
            results_df[dataset]["actual_answer"] = results_df[dataset]["actual_answer"].astype(str)
    results_1
    return dataset, results_1, results_df


@app.cell
def __(datasets, df_1, df_2, pd, test_results_dir):
    results_2 = {index: {dataset: pd.read_csv(f"{test_results_dir}/run_2/{dataset}_pipeline_{index - max(df_1["index"]) - 1}_responses.csv") for dataset in datasets} for index in df_2["index"]}
    for results_df_ in results_2.values():
        for dataset_ in datasets:
            results_df_[dataset_]["actual_answer"] = results_df_[dataset_]["actual_answer"].astype(str)
    results_2
    return dataset_, results_2, results_df_


@app.cell
def __(df_1, results_1, results_2):
    results = {index: result for index, result in results_1.items()}
    for index, result in results_2.items():
        results[index+max(df_1["index"]) + 1] = result
    results
    return index, result, results


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
def __(df):
    temp_top_p_top_k_col = df["temperature"].astype(str).add(",").add(df["top_p"].astype(str)).add(",").add(df["top_k"].astype(str))
    temp_top_p_top_k_col
    return (temp_top_p_top_k_col,)


@app.cell
def __(temp_top_p_top_k_col):
    for x in temp_top_p_top_k_col:
        print(x)
    return (x,)


@app.cell
def __(
    cosine_similarity_scores_with_average,
    datasets,
    sns,
    temp_top_p_top_k_col,
):
    ax = sns.heatmap(cosine_similarity_scores_with_average, yticklabels=temp_top_p_top_k_col, xticklabels=datasets + ["average"], cbar_kws={'label': 'Average cosine similarity'})
    ax.set(xlabel="Dataset", ylabel="Temperature,top-p,top-k")
    return (ax,)


if __name__ == "__main__":
    app.run()
