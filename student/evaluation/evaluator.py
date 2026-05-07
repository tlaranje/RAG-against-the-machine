from student.models import StudentSearchResults, RagDataset  # AnsweredQuestion
# from rich import print
import json


class Evaluator:
    def evaluate(
        self, answer_path: str, dataset_path: str, k: int = 1
    ) -> None:
        try:
            with open(answer_path, "r") as fd:
                answers = json.load(fd)
            with open(dataset_path, "r") as fd:
                dataset = json.load(fd)

            StudentSearchResults.model_validate(answers)
            RagDataset.model_validate(dataset)

            # correct_by_id = {
            #     q.question_id: q
            #     for q in b.rag_questions
            #     if isinstance(q, AnsweredQuestion)
            # }

            # total_recall = 0.0
            # total_questions = 0

        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {answer_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format in '{answer_path}': {e}")
        except Exception as e:
            raise e
