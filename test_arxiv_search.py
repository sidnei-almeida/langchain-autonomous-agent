"""Unit tests for arXiv search pipeline (no network required)."""
import unittest

from arxiv_search import (
    ARXIV_FETCH_MAX,
    MIN_RELEVANCE_SCORE,
    PaperCandidate,
    build_arxiv_queries,
    extract_paper_topic,
    is_ambiguous_query,
    is_valid_paper_topic,
    resolve_arxiv_topic,
    score_paper_relevance,
    search_scientific_papers_structured,
)

TWEAC = PaperCandidate(
    title="TWEAC: Transformer with Extendable QA Agent Classifiers",
    authors=["Author A"],
    year=2021,
    url="https://arxiv.org/abs/2104.07081",
    pdf_url="https://arxiv.org/pdf/2104.07081",
    summary=(
        "We propose TWEAC, a transformer architecture for question answering "
        "with extendable QA agent classifiers and routing between agents."
    ),
    categories=["cs.CL", "cs.AI"],
    arxiv_id="2104.07081",
)

HCI_PAPER = PaperCandidate(
    title="Designing Human-Computer Interaction for Collaborative Workspaces",
    authors=["Jane Doe"],
    year=2023,
    url="https://arxiv.org/abs/2301.00001",
    pdf_url="",
    summary=(
        "This paper studies human-computer interaction patterns, usability, "
        "and user interaction design in collaborative interfaces."
    ),
    categories=["cs.HC"],
    arxiv_id="2301.00001",
)

HUMAN_AI_PAPER = PaperCandidate(
    title="Human-AI Collaboration in Interactive Decision Support",
    authors=["John Smith"],
    year=2024,
    url="https://arxiv.org/abs/2401.00002",
    pdf_url="",
    summary=(
        "We examine human-AI interaction and human-centered AI systems "
        "for joint human-AI collaboration tasks."
    ),
    categories=["cs.AI", "cs.HC"],
    arxiv_id="2401.00002",
)


class TestAmbiguity(unittest.TestCase):
    def test_human_interactions_is_ambiguous(self):
        self.assertTrue(is_ambiguous_query("human interactions"))
        self.assertTrue(is_ambiguous_query("Human Interactions"))

    def test_hci_is_not_ambiguous(self):
        self.assertFalse(is_ambiguous_query("human-computer interaction"))

    def test_build_queries_clarification(self):
        queries, intent, needs = build_arxiv_queries("human interactions")
        self.assertTrue(needs)
        self.assertEqual(queries, [])

    def test_build_queries_hci(self):
        queries, intent, needs = build_arxiv_queries("human-computer interaction")
        self.assertFalse(needs)
        self.assertEqual(intent, "hci")
        self.assertIn("human-computer interaction", queries)


class TestScoring(unittest.TestCase):
    def test_tweac_low_score_for_human_interactions(self):
        score, _ = score_paper_relevance(TWEAC, "human interactions", "general")
        self.assertLess(score, MIN_RELEVANCE_SCORE)

    def test_tweac_low_score_for_hci_intent(self):
        score, _ = score_paper_relevance(TWEAC, "human-computer interaction", "hci")
        self.assertLess(score, MIN_RELEVANCE_SCORE)

    def test_tweac_ok_for_qa_query(self):
        score, _ = score_paper_relevance(
            TWEAC, "QA agent classifiers question answering", "qa"
        )
        self.assertGreaterEqual(score, MIN_RELEVANCE_SCORE)

    def test_hci_paper_scores_high_for_hci(self):
        score, _ = score_paper_relevance(
            HCI_PAPER, "human-computer interaction", "hci"
        )
        self.assertGreaterEqual(score, MIN_RELEVANCE_SCORE)

    def test_human_ai_paper_for_human_ai_query(self):
        score, _ = score_paper_relevance(
            HUMAN_AI_PAPER, "human-AI interaction", "human_ai"
        )
        self.assertGreaterEqual(score, MIN_RELEVANCE_SCORE)


class TestExtraction(unittest.TestCase):
    def test_portuguese_paper_request(self):
        topic = extract_paper_topic(
            "me manda um paper sobre human interactions"
        )
        self.assertIsNotNone(topic)
        self.assertEqual(topic, "human interactions")

    def test_english_paper_request(self):
        topic = extract_paper_topic(
            "find recent papers about human-AI interaction"
        )
        self.assertIsNotNone(topic)
        self.assertIn("human", topic.lower())

    def test_invalid_topic_paper_only(self):
        self.assertFalse(is_valid_paper_topic("paper"))

    def test_resolve_ignores_bad_llm_rewrite(self):
        topic = resolve_arxiv_topic(
            "me manda um paper sobre human interactions",
            "paper",
        )
        self.assertEqual(topic, "human interactions")

    def test_resolve_clarification_for_ambiguous(self):
        r = search_scientific_papers_structured(
            resolve_arxiv_topic(
                "me manda um paper sobre human interactions",
                "paper",
            )
        )
        self.assertEqual(r["type"], "clarification")


class TestStructuredClarification(unittest.TestCase):
    def test_ambiguous_returns_clarification_without_network(self):
        result = search_scientific_papers_structured("human interactions")
        self.assertEqual(result["type"], "clarification")
        self.assertTrue(result["needsClarification"])
        self.assertEqual(len(result["papers"]), 0)


class TestConstants(unittest.TestCase):
    def test_fetch_max_at_least_10(self):
        self.assertGreaterEqual(ARXIV_FETCH_MAX, 10)

    def test_threshold_is_6(self):
        self.assertEqual(MIN_RELEVANCE_SCORE, 6)


if __name__ == "__main__":
    unittest.main()
