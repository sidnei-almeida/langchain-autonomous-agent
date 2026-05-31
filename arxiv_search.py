"""
ArXiv search pipeline: ambiguity detection, query expansion, relevance scoring, filtering.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import arxiv

ARXIV_FETCH_MAX = 12
MIN_RELEVANCE_SCORE = 6
MAX_RETURN_PAPERS = 3
MAX_RETURN_PAPERS_DEEP = 5
RECENT_YEAR_CUTOFF = 2023  # papers from this year onward when "recent" requested

STOPWORDS = frozenset(
    {
        "a", "an", "the", "about", "on", "for", "of", "me", "um", "uma", "um",
        "paper", "papers", "artigo", "artigos", "find", "get", "send", "manda",
        "envia", "recent", "latest", "some", "sobre", "research", "study",
        "studies", "please", "por", "favor",
    }
)

# Queries too broad to search without clarification
AMBIGUOUS_PHRASES = frozenset(
    {
        "human interactions",
        "human interaction",
        "human behaviors",
        "human behavior",
        "interactions",
        "interaction",
        "ai",
        "cancer",
        "climate",
        "robots",
        "robot",
    }
)

VAGUE_SINGLE_TERMS = frozenset({"ai", "cancer", "climate", "robots", "robot"})

CLARIFICATION_OPTIONS_HUMAN_INTERACTION = [
    "Human-computer interaction",
    "Human-AI interaction",
    "Human-robot interaction",
    "Social/behavioral interaction",
]

CLARIFICATION_MESSAGE_HUMAN_INTERACTION = (
    "Human interactions is broad. Do you mean human-computer interaction, "
    "human-AI interaction, human-robot interaction, or social/behavioral interaction?"
)

INTENT_CATEGORY_PRIORITY = {
    "hci": {"cs.HC": 5, "cs.AI": 1, "cs.RO": 0, "cs.CL": -2, "cs.CY": 1},
    "human_ai": {"cs.AI": 4, "cs.HC": 3, "cs.CL": 1, "cs.RO": 0, "cs.CY": 2},
    "hri": {"cs.RO": 5, "cs.AI": 2, "cs.HC": 1, "cs.CL": -2, "cs.CY": 0},
    "social": {"cs.CY": 4, "cs.HC": 3, "cs.AI": 1, "cs.CL": 0, "cs.RO": 0},
    "qa": {"cs.CL": 4, "cs.AI": 3, "cs.LG": 3, "cs.HC": -1},
    "general": {"cs.AI": 1, "cs.CL": 1, "cs.HC": 1, "cs.RO": 1, "cs.CY": 1},
}

QA_MISMATCH_SIGNALS = (
    "question answering",
    "qa agent",
    "qa agents",
    "classifier",
    "classifiers",
    "tweac",
    "extendable qa",
    "routing agent",
    "transformer with extendable",
)

HUMAN_INTERACTION_SIGNALS = (
    "human",
    "interaction",
    "interactions",
    "user",
    "hci",
    "human-computer",
    "human-robot",
    "human-ai",
    "human-robot interaction",
    "social",
    "behavioral",
    "collaboration",
    "usability",
    "interface",
)


@dataclass
class PaperCandidate:
    title: str
    authors: list[str]
    year: int
    url: str
    pdf_url: str
    summary: str
    categories: list[str]
    arxiv_id: str
    relevance_score: int = 0
    why_it_matches: str = ""


def normalize_query(query: str) -> str:
    q = re.sub(r"\s+", " ", (query or "").strip().lower())
    q = re.sub(r"[^\w\s\-/]", " ", q)
    return re.sub(r"\s+", " ", q).strip()


def content_words(query: str) -> list[str]:
    return [w for w in normalize_query(query).split() if w not in STOPWORDS and len(w) > 1]


# Too vague to use as an arXiv topic alone
INVALID_TOPICS = frozenset(
    {
        "paper",
        "papers",
        "artigo",
        "artigos",
        "research",
        "study",
        "studies",
        "estudo",
        "estudos",
        "arxiv",
        "um",
        "uma",
        "a",
    }
)


def _clean_extracted_topic(topic: str) -> str:
    """Strip filler prefixes left after regex capture."""
    t = re.sub(r"[?.!]+$", "", topic.strip())
    t = re.sub(
        r"^(?:um|uma|a|the)\s+(?:paper|papers|artigo|artigos)\s+(?:sobre|about|on)\s+",
        "",
        t,
        flags=re.IGNORECASE,
    )
    t = re.sub(
        r"^(?:paper|papers|artigo|artigos)\s+(?:sobre|about|on|regarding|acerca)\s+",
        "",
        t,
        flags=re.IGNORECASE,
    )
    return t.strip()


def is_valid_paper_topic(topic: str) -> bool:
    if not topic or len(topic.strip()) < 3:
        return False
    norm = normalize_query(topic)
    if norm in INVALID_TOPICS:
        return False
    words = content_words(topic)
    if not words:
        return False
    if len(words) == 1 and words[0] in INVALID_TOPICS:
        return False
    return True


def extract_paper_topic(message: str) -> str | None:
    """Extract research topic from a user message requesting papers."""
    if not message:
        return None
    text = message.strip()
    lower = text.lower()

    if not any(
        kw in lower
        for kw in (
            "paper",
            "papers",
            "artigo",
            "artigos",
            "arxiv",
            "research",
            "estudo",
            "estudos",
            "study",
            "studies",
        )
    ):
        return None

    # Portuguese / explicit patterns first (avoid greedy optional groups)
    patterns = [
        r"(?:me\s+)?(?:manda|envia|mande|envie|send)\s+(?:me\s+)?(?:um\s+)?(?:paper|papers|artigo|artigos)\s+(?:sobre|about|on|regarding|acerca)\s+(.+)",
        r"(?:quero|preciso\s+de)\s+(?:um\s+)?(?:paper|papers|artigo|artigos)\s+(?:sobre|about|on)\s+(.+)",
        r"(?:paper|papers|artigo|artigos|study|studies|research)\s+(?:sobre|about|on|regarding|acerca)\s+(.+)",
        r"(?:find|get|search|busca|buscar|give)\s+(?:me\s+)?(?:um\s+)?(?:a\s+)?(?:paper|papers|artigo|artigos)\s+(?:sobre|about|on|regarding)\s+(.+)",
        r"(?:find|get|search)\s+(?:for\s+)?(?:papers?|articles?|studies)\s+(?:on|about|regarding)\s+(.+)",
        r"arxiv\s+(?:search\s+)?(?:for\s+)?(.+)",
        r"(?:research|papers|articles|studies)\s+(?:about|on|that say|that affirm)\s+(.+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, lower, re.IGNORECASE)
        if match:
            topic = _clean_extracted_topic(match.group(1))
            if is_valid_paper_topic(topic):
                return topic

    return None


def resolve_arxiv_topic(user_message: str, query_rewrite: str | None = None) -> str:
    """
    Best topic for arXiv: prefer extraction from original user text.
    Ignores LLM rewrites that collapse to 'paper' or other invalid tokens.
    """
    for source in (user_message, query_rewrite or ""):
        if not source:
            continue
        extracted = extract_paper_topic(source)
        if extracted and is_valid_paper_topic(extracted):
            return extracted

    # Full user message may still contain the topic after "sobre/about"
    for source in (user_message, query_rewrite or ""):
        match = re.search(
            r"(?:sobre|about|on|regarding|acerca)\s+(.+)$",
            source.strip(),
            re.IGNORECASE,
        )
        if match:
            topic = _clean_extracted_topic(match.group(1))
            if is_valid_paper_topic(topic):
                return topic

    fallback = (query_rewrite or user_message).strip()
    if is_valid_paper_topic(fallback):
        return fallback
    return user_message.strip()


def detect_intent(query: str) -> str:
    q = normalize_query(query)
    if any(
        term in q
        for term in (
            "human-computer",
            "human computer",
            "hci",
            "interaction design",
            "cs.hc",
            "user interaction",
        )
    ):
        return "hci"
    if any(
        term in q
        for term in (
            "human-ai",
            "human ai",
            "human-centered ai",
            "human centered ai",
            "human-ai collaboration",
            "human ai collaboration",
        )
    ):
        return "human_ai"
    if any(
        term in q
        for term in (
            "human-robot",
            "human robot",
            "hri",
            "robotics interaction",
            "human robot interaction",
        )
    ):
        return "hri"
    if any(
        term in q
        for term in (
            "social interaction",
            "behavioral interaction",
            "social/behavioral",
            "social behavioral",
        )
    ):
        return "social"
    if any(
        term in q
        for term in (
            "question answering",
            "qa agent",
            "qa agents",
            "qa classifier",
            "qa routing",
            "tweac",
            "transformer qa",
        )
    ):
        return "qa"
    return "general"


def is_ambiguous_query(query: str) -> bool:
    q = normalize_query(query)
    if not q:
        return True

    if q in AMBIGUOUS_PHRASES:
        return True

    if re.fullmatch(r"human\s+interactions?", q):
        return True

    words = content_words(query)
    if len(words) <= 1 and q in VAGUE_SINGLE_TERMS:
        return True

    if len(words) <= 2 and detect_intent(query) == "general":
        if q in AMBIGUOUS_PHRASES or any(p in q for p in ("human interaction", "human behavior")):
            return True

    return False


def build_arxiv_queries(user_query: str) -> tuple[list[str], str, bool]:
    """
    Build one or more arXiv search strings.
    Returns (queries, intent, needs_clarification).
    """
    if is_ambiguous_query(user_query):
        return [], detect_intent(user_query), True

    intent = detect_intent(user_query)
    q = normalize_query(user_query)

    if intent == "hci":
        return (
            [
                "human-computer interaction",
                "HCI user interaction",
                "interaction design usability",
            ],
            intent,
            False,
        )
    if intent == "human_ai":
        return (
            [
                "human AI interaction",
                "human-centered AI",
                "human-AI collaboration",
            ],
            intent,
            False,
        )
    if intent == "hri":
        return (
            [
                "human robot interaction",
                "HRI human-robot interaction",
                "robotics social interaction",
            ],
            intent,
            False,
        )
    if intent == "social":
        return (
            [
                "social interaction behavioral",
                "human social behavior interaction",
            ],
            intent,
            False,
        )
    if intent == "qa":
        return (
            [
                user_query,
                "question answering agent classifier",
                "QA agent routing transformer",
            ],
            intent,
            False,
        )

    return ([user_query, f"all:{user_query}"], intent, False)


def _important_terms(query: str) -> list[str]:
    terms = content_words(query)
    phrases = []
    q = normalize_query(query)
    for phrase in (
        "human-computer interaction",
        "human-computer",
        "human-ai",
        "human-robot",
        "question answering",
        "qa agent",
    ):
        if phrase in q:
            phrases.append(phrase)
    return terms + phrases


def score_paper_relevance(
    paper: PaperCandidate,
    user_query: str,
    intent: str,
) -> tuple[int, str]:
    score = 0
    reasons: list[str] = []
    title = paper.title.lower()
    abstract = paper.summary.lower()
    combined = f"{title} {abstract}"
    query_norm = normalize_query(user_query)
    terms = _important_terms(user_query)

    if query_norm and query_norm in title:
        score += 5
        reasons.append("exact query phrase in title")
    elif query_norm and query_norm in abstract:
        score += 3
        reasons.append("exact query phrase in abstract")

    for term in terms:
        if len(term) < 3:
            continue
        if term in title:
            score += 3
            reasons.append(f"'{term}' in title")
        elif term in abstract:
            score += 1
            reasons.append(f"'{term}' in abstract")

    intent_phrases = {
        "hci": ("human-computer interaction", "human computer interaction", "hci", "usability", "interaction design"),
        "human_ai": ("human-ai", "human ai", "human-centered", "human-ai collaboration"),
        "hri": ("human-robot", "human robot", "hri", "human-robot interaction"),
        "social": ("social interaction", "behavioral", "social behavior"),
        "qa": ("question answering", "qa agent", "classifier", "routing"),
    }
    for phrase in intent_phrases.get(intent, ()):
        if phrase in combined:
            score += 4
            reasons.append(f"intent phrase '{phrase}'")

    cat_weights = INTENT_CATEGORY_PRIORITY.get(intent, INTENT_CATEGORY_PRIORITY["general"])
    for cat in paper.categories:
        prefix = cat.split(".")[0] if "." in cat else cat
        full = cat if cat.startswith("cs.") else f"cs.{prefix}" if prefix in ("HC", "AI", "RO", "CL", "CY", "LG") else cat
        for key, weight in cat_weights.items():
            if cat == key or cat.startswith(key):
                score += weight
                if weight >= 3:
                    reasons.append(f"category {cat}")
                break

    if intent in ("hci", "human_ai", "hri", "social"):
        has_human = any(s in combined for s in HUMAN_INTERACTION_SIGNALS)
        has_qa_only = any(s in combined for s in QA_MISMATCH_SIGNALS)
        if has_qa_only and not has_human:
            score -= 6
            reasons.append("penalty: QA/classifier focus without human interaction")
        if not has_human and "interaction" not in combined and "human" in query_norm:
            score -= 4
            reasons.append("penalty: missing human/interaction in paper")

    if intent == "hci" and "cs.CL" in paper.categories and "cs.HC" not in paper.categories:
        if not any(t in combined for t in ("human", "user", "interface", "hci", "interaction")):
            score -= 3
            reasons.append("penalty: cs.CL without HCI context")

    if intent == "qa":
        if any(s in combined for s in QA_MISMATCH_SIGNALS):
            score += 3

    why = (
        "; ".join(reasons[:4])
        if reasons
        else "Limited term overlap with query"
    )
    return score, why


def _result_to_candidate(result: arxiv.Result) -> PaperCandidate:
    authors = [a.name for a in result.authors]
    abstract = result.summary.replace("\n", " ").strip()
    arxiv_id = result.entry_id.split("/abs/")[-1] if "/abs/" in result.entry_id else result.entry_id
    return PaperCandidate(
        title=result.title,
        authors=authors,
        year=result.published.year,
        url=result.entry_id,
        pdf_url=result.pdf_url or "",
        summary=abstract[:600] + ("..." if len(abstract) > 600 else ""),
        categories=list(result.categories),
        arxiv_id=arxiv_id,
    )


def wants_recent_papers(query: str) -> bool:
    q = normalize_query(query)
    return any(
        kw in q
        for kw in (
            "latest", "recent", "current", "new", "newest",
            "2024", "2025", "2026", "last year", "this year",
        )
    )


def fetch_arxiv_candidates(queries: list[str], max_per_query: int = ARXIV_FETCH_MAX) -> list[PaperCandidate]:
    client = arxiv.Client()
    seen: set[str] = set()
    candidates: list[PaperCandidate] = []

    for query in queries:
        search = arxiv.Search(
            query=query,
            max_results=max_per_query,
            sort_by=arxiv.SortCriterion.Relevance,
        )
        try:
            for result in client.results(search):
                paper = _result_to_candidate(result)
                if paper.arxiv_id in seen:
                    continue
                seen.add(paper.arxiv_id)
                candidates.append(paper)
        except Exception:
            continue

    return candidates


def clarification_payload(message: str, options: list[str]) -> dict[str, Any]:
    return {
        "type": "clarification",
        "message": message,
        "options": options,
        "needsClarification": True,
        "papers": [],
    }


def paper_results_payload(
    message: str,
    papers: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "type": "paper_results",
        "message": message,
        "papers": papers,
        "needsClarification": False,
    }


def _filter_by_year(candidates: list[PaperCandidate], min_year: int | None) -> list[PaperCandidate]:
    if min_year is None:
        return candidates
    filtered = [p for p in candidates if p.year >= min_year]
    return filtered if filtered else candidates


def search_scientific_papers_structured(
    user_query: str,
    *,
    max_papers: int | None = None,
    recent_only: bool = False,
) -> dict[str, Any]:
    """
    Main pipeline: ambiguity check → query expansion → fetch → score → filter.
    """
    topic = user_query.strip()
    limit = max_papers or MAX_RETURN_PAPERS
    if limit > MAX_RETURN_PAPERS_DEEP:
        limit = MAX_RETURN_PAPERS_DEEP

    if not topic:
        return clarification_payload(
            "What topic should I search for on arXiv?",
            CLARIFICATION_OPTIONS_HUMAN_INTERACTION,
        )

    queries, intent, needs_clarification = build_arxiv_queries(topic)

    if needs_clarification:
        if "human" in normalize_query(topic) and "interaction" in normalize_query(topic):
            return clarification_payload(
                CLARIFICATION_MESSAGE_HUMAN_INTERACTION,
                CLARIFICATION_OPTIONS_HUMAN_INTERACTION,
            )
        return clarification_payload(
            f"'{topic}' is too broad for a precise arXiv search. "
            "Please narrow the topic (e.g., human-computer interaction, human-AI interaction, "
            "or a specific disease/mechanism).",
            CLARIFICATION_OPTIONS_HUMAN_INTERACTION,
        )

    candidates = fetch_arxiv_candidates(queries)
    min_year = RECENT_YEAR_CUTOFF if recent_only or wants_recent_papers(topic) else None
    candidates = _filter_by_year(candidates, min_year)
    if not candidates:
        return clarification_payload(
            f"I did not find arXiv results for '{topic}'. "
            "Try a more specific phrase or clarify the research area.",
            CLARIFICATION_OPTIONS_HUMAN_INTERACTION,
        )

    scored: list[tuple[PaperCandidate, int, str]] = []
    for paper in candidates:
        rel_score, why = score_paper_relevance(paper, topic, intent)
        paper.relevance_score = rel_score
        paper.why_it_matches = why
        scored.append((paper, rel_score, why))

    scored.sort(key=lambda x: x[1], reverse=True)
    strong = [p for p, s, _ in scored if s >= MIN_RELEVANCE_SCORE]

    if not strong:
        msg = f"I did not find a strong arXiv match for '{topic}'."
        if min_year:
            msg += f" No strong matches since {min_year}."
        msg += (
            " Try a more specific phrase or clarify the research area."
        )
        return clarification_payload(msg, CLARIFICATION_OPTIONS_HUMAN_INTERACTION)

    top = strong[:limit]
    papers_out = [
        {
            "title": p.title,
            "authors": p.authors[:5],
            "year": p.year,
            "url": p.url,
            "pdfUrl": p.pdf_url,
            "summary": p.summary,
            "categories": p.categories,
            "relevanceScore": p.relevance_score,
            "whyItMatches": p.why_it_matches,
        }
        for p in top
    ]

    intent_label = {
        "hci": "human-computer interaction",
        "human_ai": "human-AI interaction",
        "hri": "human-robot interaction",
        "social": "social/behavioral interaction",
        "qa": "question answering / QA agents",
    }.get(intent, topic)

    message = (
        f"I found {len(top)} paper(s) with strong relevance to {intent_label}."
        if intent != "general"
        else f"I found {len(top)} paper(s) matching '{topic}'."
    )

    return paper_results_payload(message, papers_out)


def format_structured_result_for_agent(result: dict[str, Any]) -> str:
    """Format structured search result as context for the LLM or direct user reply."""
    if result.get("type") == "clarification":
        options = result.get("options") or []
        opts = "\n".join(f"  - {o}" for o in options)
        return (
            f"[ArXiv search — clarification needed]\n"
            f"{result['message']}\n\n"
            f"Options:\n{opts}\n\n"
            f"Do not recommend papers until the user clarifies."
        )

    lines = [
        f"[ArXiv search — ranked results]\n{result.get('message', '')}\n",
        "Only cite papers listed below. Do not add papers not in this list.",
        "Explain relevance using title/abstract only — do not invent matches.\n",
    ]
    for i, paper in enumerate(result.get("papers") or [], 1):
        authors = ", ".join(paper.get("authors") or [])[:120]
        lines.append(
            f"--- Paper {i} (relevance score: {paper.get('relevanceScore', 0)}) ---\n"
            f"Title: {paper.get('title')}\n"
            f"Authors: {authors}\n"
            f"Year: {paper.get('year')}\n"
            f"URL: {paper.get('url')}\n"
            f"Categories: {', '.join(paper.get('categories') or [])}\n"
            f"Why it matches: {paper.get('whyItMatches')}\n"
            f"Abstract: {paper.get('summary')}\n"
        )
    return "\n".join(lines)


def search_scientific_papers(query: str) -> str:
    """Tool entry point: returns formatted string for agent context."""
    result = search_scientific_papers_structured(query)
    return format_structured_result_for_agent(result)
