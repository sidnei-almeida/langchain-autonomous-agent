"""Gray Matter Research Agent — agentic scientific research pipeline."""

from agent.graph import GrayMatterResearchAgent, create_scientific_agent
from agent.messages import prepare_messages

# Backward-compatible alias
SimpleScientificAgent = GrayMatterResearchAgent

__all__ = [
    "GrayMatterResearchAgent",
    "SimpleScientificAgent",
    "create_scientific_agent",
    "prepare_messages",
]
