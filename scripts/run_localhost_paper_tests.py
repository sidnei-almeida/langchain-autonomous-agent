#!/usr/bin/env python3
"""Test paper-search behavior against local API."""
import json
import sys
import time
import urllib.error
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:7860"

QUESTIONS = [
    ("me manda um paper sobre human interactions", "clarification", False),
    ("me manda um paper sobre human-computer interaction", "papers", False),
    ("find recent papers about human-AI interaction", "papers", False),
    ("paper about QA agent classifiers", "papers", True),  # TWEAC ok
    ("paper about transformer QA routing agents", "papers", True),
    ("find research about CRISPR delivery methods", "papers", False),
    ("Explain lung cancer risk factors", "general", False),
    ("Who in Breaking Bad had cancer?", "general", False),
]


def post_query(question: str) -> dict:
    body = json.dumps({"question": question}).encode()
    req = urllib.request.Request(
        f"{BASE}/api/query",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        return json.loads(resp.read().decode())


def main() -> None:
    print(f"API base: {BASE}\n")
    results = []

    for question, kind, tweac_ok in QUESTIONS:
        print("=" * 72)
        print(f"Q: {question}")
        t0 = time.time()
        try:
            data = post_query(question)
        except urllib.error.HTTPError as e:
            print(f"HTTP {e.code}: {e.read().decode()[:300]}")
            results.append((question, "ERRO HTTP"))
            continue
        except Exception as e:
            print(f"Erro: {e}")
            results.append((question, "ERRO"))
            continue

        elapsed = round(time.time() - t0, 1)
        answer = data.get("answer", "")
        tools = data.get("tools_used")
        has_tweac = "tweac" in answer.lower() or "2104.07081" in answer
        is_clarify = "broad" in answer.lower() or "do you mean" in answer.lower()
        has_arxiv = "arxiv.org" in answer.lower()

        ok = True
        if kind == "clarification":
            ok = is_clarify and not has_tweac
        elif kind == "papers":
            ok = has_arxiv or is_clarify  # clarify ok if no match
            if not tweac_ok and has_tweac:
                ok = False

        status = "OK" if ok else "REVISAR"
        results.append((question, status))

        print(f"[{status}] {elapsed}s | tools={tools}")
        print(f"  TWEAC={has_tweac} | clarificação={is_clarify} | arxiv_url={has_arxiv}")
        print(f"  Resposta:\n{answer[:600]}")
        if len(answer) > 600:
            print("  ...")

    print("\n" + "=" * 72)
    print("RESUMO")
    ok_count = sum(1 for _, s in results if s == "OK")
    for q, s in results:
        print(f"  [{s}] {q[:60]}")
    print(f"\n{ok_count}/{len(results)} OK")


if __name__ == "__main__":
    main()
