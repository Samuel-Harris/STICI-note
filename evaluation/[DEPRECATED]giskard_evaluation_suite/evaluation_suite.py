import giskard
import pandas as pd
from giskard import Dataset, TestResult
from giskard.testing.tests.llm.ground_truth import test_llm_ground_truth_similarity

from evaluation.evaluation_configs.evaluation_configs import generate_test_pipeline_df, TestPipeline

from evaluation.pipelines.pipeline import Pipeline


test_df = pd.read_csv("../data/single_passage_test_questions.csv")


def predict(pipeline: Pipeline, df: pd.DataFrame):
    return [pipeline.query(question) for question in df["question"]]


test_pipeline: TestPipeline
for test_pipeline in generate_test_pipeline_df():
    giskard_model: giskard.Model = giskard.Model(
        model=lambda df: predict(test_pipeline.pipeline, df),
        model_type="text_generation",
        name="Climate Change Question Answering",
        description="This model answers any question about climate change based on IPCC reports",
        feature_names=["question"],
    )
    dataset = Dataset(test_df, target="answer")
    result: TestResult = test_llm_ground_truth_similarity(giskard_model, dataset).execute()
    print(result)
    print(result.details)
