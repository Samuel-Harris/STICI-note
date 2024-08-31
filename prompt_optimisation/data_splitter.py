import pandas as pd
import random

document_df = pd.read_csv("../data/documents.csv")
no_answer_questions_df = pd.read_csv("../data/no_answer_questions.csv")
single_passage_df = pd.read_csv("../data/single_passage_answer_questions.csv")
multi_passage_df = pd.read_csv("../data/multi_passage_answer_questions.csv")

indexes = list(document_df["index"])
random.shuffle(indexes)
cutoff_index = len(indexes) // 2
validate_indexes = indexes[:cutoff_index]
test_indexes = indexes[cutoff_index:]

no_answer_validate_questions_df = no_answer_questions_df[no_answer_questions_df["document_index"].isin(validate_indexes)]
no_answer_test_questions_df = no_answer_questions_df[no_answer_questions_df["document_index"].isin(test_indexes)]
no_answer_validate_questions_df.to_csv("../data/no_answer_validate_questions.csv", index=False)
no_answer_test_questions_df.to_csv("../data/no_answer_test_questions.csv", index=False)

single_passage_validate_questions_df = single_passage_df[single_passage_df["document_index"].isin(validate_indexes)]
single_passage_test_questions_df = single_passage_df[single_passage_df["document_index"].isin(test_indexes)]
single_passage_validate_questions_df.to_csv("../data/single_passage_validate_questions.csv", index=False)
single_passage_test_questions_df.to_csv("../data/single_passage_test_questions.csv", index=False)

multi_passage_validate_questions_df = multi_passage_df[multi_passage_df["document_index"].isin(validate_indexes)]
multi_passage_test_questions_df = multi_passage_df[multi_passage_df["document_index"].isin(test_indexes)]
multi_passage_validate_questions_df.to_csv("../data/multi_passage_validate_questions.csv", index=False)
multi_passage_test_questions_df.to_csv("../data/multi_passage_test_questions.csv", index=False)

