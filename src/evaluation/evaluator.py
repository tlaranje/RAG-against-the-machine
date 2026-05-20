from src.models import StudentSearchResults, RagDataset, AnsweredQuestion
import json


class Evaluator:
    def evaluate(
        self, results_path: str, answers_path: str, k: int = 1
    ) -> float:
        """
        Compute mean recall@k over all answered questions.

        Args:
            results_path: Path to the JSON file with search results.
            answers_path: Path to the JSON file with ground-truth answers.
            k: Number of top retrieved sources to consider.

        Returns:
            Mean recall in [0.0, 1.0]. Returns 0.0 if no questions found.

        Raises:
            FileNotFoundError: If either file does not exist.
            ValueError: If either file contains invalid JSON.
        """
        try:
            # Load raw JSON from disk for both results and answers.
            with open(results_path, "r") as fd:
                raw_results = json.load(fd)
            with open(answers_path, "r") as fd:
                raw_answers = json.load(fd)

            # Parse and validate the JSON payloads into typed models.
            results = StudentSearchResults.model_validate(raw_results)
            answers = RagDataset.model_validate(raw_answers)

            # Build a lookup from question_id to its AnsweredQuestion so
            # we can match each result to its ground-truth in O(1).
            correct_by_id = {
                q.question_id: q
                for q in answers.rag_questions
                if isinstance(q, AnsweredQuestion)
            }

            # Accumulators for computing the mean recall at the end.
            total_recall = 0.0
            total_questions = 0

            for result in results.search_results:
                # Skip results that have no matching ground-truth entry.
                correct = correct_by_id.get(result.question_id)
                if not correct:
                    continue

                # Count how many of the correct sources were retrieved.
                found = 0

                for correct_source in correct.sources:
                    # Only inspect the top-k retrieved sources.
                    for retrieved in result.retrieved_sources[:k]:
                        # Sources in different files can never overlap.
                        if correct_source.file_path != retrieved.file_path:
                            continue

                        # Compute character-level overlap between the
                        # retrieved chunk and the correct source passage.
                        overlap_start = max(
                            retrieved.first_character_index,
                            correct_source.first_character_index
                        )
                        overlap_end = min(
                            retrieved.last_character_index,
                            correct_source.last_character_index
                        )
                        # Clamp to zero: negative value means no overlap.
                        overlap_size = max(0, overlap_end - overlap_start)

                        # Sizes of each individual span in characters.
                        correct_size = (
                            correct_source.last_character_index
                            - correct_source.first_character_index
                        )
                        retrieved_size = (
                            retrieved.last_character_index
                            - retrieved.first_character_index
                        )

                        # Union size via inclusion-exclusion principle.
                        union_size = (
                            correct_size + retrieved_size - overlap_size
                        )

                        # Accept the chunk when IoU meets the 5% threshold.
                        if (
                            union_size > 0
                            and overlap_size / union_size >= 0.05
                        ):
                            found += 1
                            # Stop checking once a hit is found for this
                            # correct source.
                            break

                # Recall for this question: fraction of sources found.
                total_recall += found / len(correct.sources)
                total_questions += 1

            # Average recall across all evaluated questions.
            recall = (
                total_recall / total_questions
                if total_questions > 0
                else 0
            )
            return recall

        except FileNotFoundError as e:
            raise FileNotFoundError(e)
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON format")
        except Exception as e:
            raise e
