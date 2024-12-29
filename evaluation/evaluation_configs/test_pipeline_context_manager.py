import pandas as pd

from evaluation.evaluation_configs.test_pipeline import TestPipeline


class TestPipelineContextManager:
    def __init__(self, test_pipeline: TestPipeline, row: pd.Series, documents_df: pd.DataFrame) -> None:
        self.test_pipeline: TestPipeline = test_pipeline
        self.document_text: str = documents_df[documents_df["index"] == row["document_index"]]["text"].iloc[0]

    def __enter__(self) -> TestPipeline:
        self.test_pipeline.add_text_to_vector_db(self.document_text)

        return self.test_pipeline

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.test_pipeline.reset_vector_db()