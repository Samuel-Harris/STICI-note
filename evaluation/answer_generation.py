import pandas as pd
import torch
from pandas import Series
from tqdm import tqdm

from evaluation.evaluation_configs.test_pipeline import generate_test_pipeline_df, TestPipeline, \
    TestPipelineContextManager


def main() -> None:
    torch.mps.set_per_process_memory_fraction(0.0)

    documents_df: pd.DataFrame = pd.read_csv("../data/documents.csv")
    multi_passage_df: pd.DataFrame = pd.read_csv("../data/multi_passage_answer_questions.csv")
    single_passage_df: pd.DataFrame = pd.read_csv("../data/single_passage_answer_questions.csv")
    no_answer_df: pd.DataFrame = pd.read_csv("../data/no_answer_questions.csv")
    no_answer_df["expected_answer"] = ["The answer to your question is not in the provided text." for _ in
                                       range(len(no_answer_df))]

    multi_passage_df = multi_passage_df.rename(columns={"answer": "expected_answer"})
    single_passage_df = single_passage_df.rename(columns={"answer": "expected_answer"})

    multi_passage_df["actual_answer"] = [None] * len(multi_passage_df)
    single_passage_df["actual_answer"] = [None] * len(single_passage_df)
    no_answer_df["actual_answer"] = [None] * len(no_answer_df)

    test_pipeline_df: pd.DataFrame = generate_test_pipeline_df()
    test_pipeline_df.drop(columns=["test_pipeline"]).to_csv("../data/test_pipelines.csv", index_label="index")

    test_pipeline: Series
    for test_pipeline_i, test_pipeline_row in test_pipeline_df.iterrows():
        run_test_on_dataset(documents_df, multi_passage_df, test_pipeline_row.test_pipeline, "multi_passage",
                            test_pipeline_i)

        run_test_on_dataset(documents_df, single_passage_df, test_pipeline_row.test_pipeline, "single_passage",
                            test_pipeline_i)

        run_test_on_dataset(documents_df, no_answer_df, test_pipeline_row.test_pipeline, "no_answer",
                            test_pipeline_i)


def run_test_on_dataset(documents_df: pd.DataFrame, dataset: pd.DataFrame, test_pipeline: TestPipeline,
                        dataset_name: str, test_pipeline_i: int) -> None:
    print(f"Executing pipeline {test_pipeline.test_config} on {dataset_name} questions")
    progress_bar = tqdm(total=len(dataset))

    row: Series
    for question_i, row in dataset.iterrows():
        actual_answer: str = run_test_on_example(test_pipeline, row, documents_df)
        dataset.at[question_i, "actual_answer"] = actual_answer
        progress_bar.update(1)

    dataset.to_csv(f"../data/{dataset_name}_pipeline_{test_pipeline_i}_responses.csv", index_label="index")


def run_test_on_example(test_pipeline: TestPipeline, row: Series, documents_df: pd.DataFrame) -> str:
    with TestPipelineContextManager(test_pipeline, row, documents_df):
        return test_pipeline.pipeline.query(row.question)


if __name__ == "__main__":
    main()
