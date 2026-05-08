from student.models import StudentSearchResults, RagDataset, AnsweredQuestion
import json


class Evaluator:
    def evaluate(
        self, results_path: str, answers_path: str, k: int = 1
    ) -> float:
        try:
            with open(results_path, "r") as fd:
                raw_results = json.load(fd)
            with open(answers_path, "r") as fd:
                raw_answers = json.load(fd)

            results = StudentSearchResults.model_validate(raw_results)
            answers = RagDataset.model_validate(raw_answers)

            correct_by_id = {
                q.question_id: q
                for q in answers.rag_questions
                if isinstance(q, AnsweredQuestion)
            }

            total_recall = 0.0
            total_questions = 0

            for result in results.search_results:
                correct = correct_by_id.get(result.question_id)
                if not correct:
                    continue

                found = 0
                for correct_source in correct.sources:
                    for retrieved in result.retrieved_sources[:k]:
                        if correct_source.file_path != retrieved.file_path:
                            continue
                        overlap_start = max(
                            retrieved.first_character_index,
                            correct_source.first_character_index
                        )
                        overlap_end = min(
                            retrieved.last_character_index,
                            correct_source.last_character_index
                        )
                        overlap_size = max(0, overlap_end - overlap_start)
                        correct_size = (
                            correct_source.last_character_index -
                            correct_source.first_character_index
                        )
                        retrieved_size = (
                            retrieved.last_character_index -
                            retrieved.first_character_index
                        )
                        union_size = (
                            correct_size + retrieved_size - overlap_size
                        )
                        if union_size > 0 and \
                           overlap_size / union_size >= 0.05:
                            found += 1
                            break
                total_recall += found / len(correct.sources)
                total_questions += 1
            recall = (
                total_recall / total_questions if total_questions > 0 else 0
            )
            return recall
        except FileNotFoundError as e:
            raise FileNotFoundError(e)
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON format")
        except Exception as e:
            raise e
