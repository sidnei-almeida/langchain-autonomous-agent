"""Tests for Gray Matter Research Agent."""
import os
import unittest

os.environ.setdefault("GROQ_API_KEY", "test-key-for-unit-tests")


class TestIntentFallback(unittest.TestCase):
    def test_paper_search_heuristic(self):
        from agent.router import _heuristic_classify

        r = _heuristic_classify("find papers about CRISPR")
        self.assertEqual(r.intent, "paper_search")
        self.assertIn("arxiv", r.tools_required)

    def test_mixed_research_latest(self):
        from agent.router import _heuristic_classify

        r = _heuristic_classify("latest papers about agentic RAG")
        self.assertIn("arxiv", r.tools_required)
        self.assertIn("web_search", r.tools_required)

    def test_calculation_heuristic(self):
        from agent.router import _heuristic_classify

        r = _heuristic_classify("calculate 2 + 2")
        self.assertEqual(r.intent, "calculation")
        self.assertIn("calculator", r.tools_required)


class TestCalculator(unittest.TestCase):
    def test_basic_math(self):
        from agent.tools import calculator

        self.assertEqual(calculator("2 + 2"), "4")

    def test_sqrt(self):
        from agent.tools import calculator

        self.assertEqual(calculator("sqrt(16)"), "4.0")


class TestArxivAmbiguity(unittest.TestCase):
    def test_human_interactions_clarification(self):
        from arxiv_search import search_scientific_papers_structured

        r = search_scientific_papers_structured("human interactions")
        self.assertEqual(r["type"], "clarification")
        self.assertTrue(r["needsClarification"])


class TestPlanner(unittest.TestCase):
    def test_plan_has_steps(self):
        from agent.planner import build_research_plan
        from agent.state import IntentResult

        intent = IntentResult(
            intent="mixed_research",
            tools_required=["arxiv", "web_search"],
        )
        plan = build_research_plan(intent, "latest agentic RAG papers")
        self.assertGreaterEqual(len(plan), 2)
        self.assertLessEqual(len(plan), 5)


class TestEvidence(unittest.TestCase):
    def test_rank_and_cap(self):
        from agent.evidence import rank_evidence
        from agent.state import EvidenceItem, ResearchState

        state = ResearchState(
            user_query="test",
            max_sources=2,
            evidence=[
                EvidenceItem("A", "", "web", "a", 0.5),
                EvidenceItem("B", "", "web", "b", 0.9),
                EvidenceItem("C", "", "web", "c", 0.7),
            ],
        )
        state = rank_evidence(state)
        self.assertEqual(len(state.evidence), 2)
        self.assertEqual(state.evidence[0].title, "B")


class TestAPIHealth(unittest.TestCase):
    def test_health_endpoint(self):
        from fastapi.testclient import TestClient
        from api import app

        client = TestClient(app)
        r = client.get("/health")
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertIn("status", data)
        self.assertIn("agent_initialized", data)


class TestStructuredResponseShape(unittest.TestCase):
    def test_build_enriched_response(self):
        from langchain_core.messages import AIMessage
        from api_helpers import build_enriched_response

        result = {
            "messages": [AIMessage(content="Test answer")],
            "tools_used": ["wikipedia"],
            "intent": "concept_explanation",
            "research_plan": ["Step 1"],
            "sources": [],
            "papers": [],
            "confidence": 0.7,
            "limitations": [],
            "follow_up_questions": [],
        }
        payload = build_enriched_response(result, question="What is DNA?")
        self.assertEqual(payload["answer"], "Test answer")
        self.assertEqual(payload["intent"], "concept_explanation")
        self.assertIn("research_plan", payload)
        self.assertIn("confidence", payload)


if __name__ == "__main__":
    unittest.main()
