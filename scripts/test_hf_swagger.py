#!/usr/bin/env python3
"""Test HF Space endpoints (same as Swagger UI)."""
import json
import sys
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "https://salmeida-langchain-agent.hf.space"


def req(method: str, path: str, body: dict | None = None, timeout: int = 120) -> dict:
    url = f"{BASE.rstrip('/')}{path}"
    data = json.dumps(body).encode() if body else None
    r = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"} if data else {},
        method=method,
    )
    with urllib.request.urlopen(r, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def show(name: str, d: dict) -> None:
    print(f"\n{'━' * 56}")
    print(f"▶ {name}")
    print("━" * 56)
    if "message" in d and "answer" not in d:
        print("content:", (d["message"].get("content") or "")[:450])
        for k in ("tools_used", "intent", "confidence", "processing_time"):
            if d.get(k) is not None:
                print(f"{k}:", d[k])
    else:
        print("answer:", (d.get("answer") or "")[:450])
        for k in ("question", "tools_used", "intent", "confidence", "processing_time", "depth"):
            if d.get(k) is not None:
                print(f"{k}:", d[k])
        if d.get("research_plan"):
            print("plan:", d["research_plan"])
    papers = d.get("papers") or []
    print("papers:", len(papers))
    for p in papers[:3]:
        print(f"  • [{p.get('relevanceScore')}] {p.get('title', '')[:58]}")
    if d.get("limitations"):
        print("limitations:", d["limitations"][:2])
    if d.get("follow_up_questions"):
        print("follow_up:", d["follow_up_questions"][0][:70])


def main() -> None:
    print(f"Base: {BASE}")
    print(f"Swagger: {BASE}/docs")

    show("GET /health", req("GET", "/health", timeout=15))
    show("GET /", req("GET", "/", timeout=15))

    tools = req("GET", "/api/tools", timeout=15)
    print(f"\n{'━' * 56}\n▶ GET /api/tools\n{'━' * 56}")
    print(json.dumps(tools, indent=2)[:400])

    show(
        "POST /api/query — human interactions (clarificação)",
        req("POST", "/api/query", {"question": "me manda um paper sobre human interactions"}, 90),
    )
    show(
        "POST /api/query — HCI",
        req("POST", "/api/query", {"question": "find papers about human-computer interaction"}, 120),
    )
    show(
        "POST /api/query — sqrt",
        req("POST", "/api/query", {"question": "What is sqrt(144)?"}, 60),
    )
    show(
        "POST /api/chat",
        req(
            "POST",
            "/api/chat",
            {"messages": [{"role": "user", "content": "Explain DNA in one sentence"}]},
            90,
        ),
    )
    show(
        "POST /api/research — deep",
        req(
            "POST",
            "/api/research",
            {
                "question": "latest advances in agentic RAG",
                "depth": "deep",
                "max_sources": 8,
            },
            180,
        ),
    )
    print(f"\n{'━' * 56}\n✓ Todos os testes OK\n{'━' * 56}")


if __name__ == "__main__":
    main()
