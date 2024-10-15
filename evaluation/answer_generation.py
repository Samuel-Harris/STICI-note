from typing import Any

import pandas as pd
import torch
from pandas import Series
from tqdm import tqdm

from evaluation.evaluation_configs.test_config_combinations import TestConfigCombinations
from evaluation.evaluation_configs.test_pipeline import TestPipeline
from evaluation.evaluation_configs.test_pipeline_context_manager import TestPipelineContextManager


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

    test_config_combinations = TestConfigCombinations()

    n_config_combinations: int = test_config_combinations.get_n_combinations()
    print(f"Testing {n_config_combinations} config combinations")

    save_test_pipeline_configs(test_config_combinations)

    for test_pipeline_i, test_pipeline in enumerate(test_config_combinations.generate_test_pipelines()):
        run_test_on_dataset(documents_df, multi_passage_df, test_pipeline, "multi_passage", test_pipeline_i,
                            n_config_combinations)

        run_test_on_dataset(documents_df, single_passage_df, test_pipeline, "single_passage", test_pipeline_i,
                            n_config_combinations)

        run_test_on_dataset(documents_df, no_answer_df, test_pipeline, "no_answer", test_pipeline_i,
                            n_config_combinations)


def save_test_pipeline_configs(test_config_combinations: TestConfigCombinations) -> None:
    rows: list[dict[str, Any]] = []
    for test_pipeline in test_config_combinations.get_test_config_combinations():
        rows.append(test_pipeline.get_attributes())
    pd.DataFrame(rows).to_csv("../data/test_pipelines.csv", index_label="index")


def run_test_on_dataset(documents_df: pd.DataFrame, dataset: pd.DataFrame, test_pipeline: TestPipeline,
                        dataset_name: str, test_pipeline_i: int, n_config_combinations: int) -> None:
    print(f"Executing pipeline {test_pipeline_i}/{n_config_combinations - 1} on {dataset_name} questions", flush=True)
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
