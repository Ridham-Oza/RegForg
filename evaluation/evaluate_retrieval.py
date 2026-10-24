import time
import statistics

from evaluation.test_questions import EVALUATION_DATASET
from src.retrieval.retriever import PineconeRetriever


TOP_K = 5


def calculate_recall_at_k(retrieved_pages, relevant_pages):
    """
    Recall@K =
    number of relevant pages retrieved /
    total number of relevant pages
    """

    retrieved = set(retrieved_pages)
    relevant = set(relevant_pages)

    if not relevant:
        return 0.0

    return len(retrieved & relevant) / len(relevant)


def calculate_hit_at_k(retrieved_pages, relevant_pages):
    """
    Hit@K = 1 if at least one relevant page
    appears in the retrieved results.
    """

    retrieved = set(retrieved_pages)
    relevant = set(relevant_pages)

    return 1.0 if retrieved & relevant else 0.0


def calculate_mrr(retrieved_pages, relevant_pages):
    """
    Mean Reciprocal Rank for a single query.
    """

    relevant = set(relevant_pages)

    for rank, page in enumerate(retrieved_pages, start=1):

        if page in relevant:
            return 1.0 / rank

    return 0.0


def evaluate():

    retriever = PineconeRetriever(
        top_k=TOP_K
    )

    recall_scores = []
    hit_scores = []
    mrr_scores = []
    similarity_scores = []
    retrieval_times = []
    end_to_end_times = []

    print("=" * 70)
    print("REGFORG — RAG RETRIEVAL EVALUATION")
    print("=" * 70)

    for i, item in enumerate(
        EVALUATION_DATASET,
        start=1
    ):

        question = item["question"]
        relevant_pages = item["relevant_pages"]

        print()
        print("-" * 70)
        print(f"TEST {i}")
        print("-" * 70)

        print("Question:")
        print(question)

        print(
            "Expected pages:",
            relevant_pages
        )

        start = time.perf_counter()

        result = retriever.retrieve(
            question
        )

        end_to_end = (
            time.perf_counter() - start
        )

        documents = result["matches"]

        retrieved_pages = [
            int(doc["page"])
            for doc in documents
        ]

        scores = [
            float(doc["score"])
            for doc in documents
        ]

        recall = calculate_recall_at_k(
            retrieved_pages,
            relevant_pages
        )

        hit = calculate_hit_at_k(
            retrieved_pages,
            relevant_pages
        )

        mrr = calculate_mrr(
            retrieved_pages,
            relevant_pages
        )

        avg_similarity = (
            statistics.mean(scores)
            if scores
            else 0.0
        )

        retrieval_time = result[
            "retrieval_time"
        ]

        recall_scores.append(recall)
        hit_scores.append(hit)
        mrr_scores.append(mrr)
        similarity_scores.append(
            avg_similarity
        )
        retrieval_times.append(
            retrieval_time
        )
        end_to_end_times.append(
            end_to_end
        )

        print()
        print(
            "Retrieved pages:",
            retrieved_pages
        )

        print(
            "Similarity scores:",
            [
                round(score, 4)
                for score in scores
            ]
        )

        print(
            "Recall@5:",
            round(recall, 4)
        )

        print(
            "Hit@5:",
            round(hit, 4)
        )

        print(
            "MRR:",
            round(mrr, 4)
        )

        print(
            "Average similarity:",
            round(avg_similarity, 4)
        )

        print(
            "Retrieval latency:",
            round(retrieval_time, 4),
            "sec"
        )

        print(
            "End-to-end retrieval latency:",
            round(end_to_end, 4),
            "sec"
        )

    # -----------------------------------------------------
    # FINAL METRICS
    # -----------------------------------------------------

    print()
    print("=" * 70)
    print("FINAL RETRIEVAL METRICS")
    print("=" * 70)

    print(
        "Recall@5              :",
        round(
            statistics.mean(recall_scores),
            4
        )
    )

    print(
        "Hit@5                 :",
        round(
            statistics.mean(hit_scores),
            4
        )
    )

    print(
        "MRR                   :",
        round(
            statistics.mean(mrr_scores),
            4
        )
    )

    print(
        "Average similarity    :",
        round(
            statistics.mean(similarity_scores),
            4
        )
    )

    print(
        "Average retrieval     :",
        round(
            statistics.mean(retrieval_times),
            4
        ),
        "sec"
    )

    print(
        "Average end-to-end    :",
        round(
            statistics.mean(end_to_end_times),
            4
        ),
        "sec"
    )

    print("=" * 70)
    print("EVALUATION COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    evaluate()
    