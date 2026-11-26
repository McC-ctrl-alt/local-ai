"""Role check after retrieval.

Every department folder under knowledge_test/ is searched. engine.py embeds
the question, keeps the cosine top 10, reranks those 10, and abstains when
the top score is below -0.7. Question wording does not choose a folder.

If every passage that survives the cutoff is from a department the role
cannot read, the answer names that rule and does not call Qwen. Otherwise
one Qwen call answers from the allowed passages only.
"""

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml

import curator
import engine


ROOT = Path(__file__).resolve().parent
POLICY_PATH = ROOT / "policy.yaml"
CORPUS_DIR = ROOT / "knowledge_test"
_RERANKER = None


def load_policy(path: Path = POLICY_PATH) -> dict[str, Any]:
    """Load roles. Each role lists departments and one named rule."""
    with open(path, encoding="utf-8") as handle:
        policy = yaml.safe_load(handle)
    roles = policy.get("roles") if isinstance(policy, dict) else None
    if not isinstance(roles, dict) or not roles:
        raise ValueError("policy.yaml must define roles.")
    for role, spec in roles.items():
        if not isinstance(spec, dict):
            raise ValueError(f"{role} must be a mapping.")
        departments = spec.get("departments")
        rule = spec.get("rule")
        if (
            not isinstance(departments, list)
            or not departments
            or any(not isinstance(item, str) or not item.strip() for item in departments)
        ):
            raise ValueError(f"{role} must list departments.")
        if not isinstance(rule, dict) or not str(rule.get("id", "")).strip():
            raise ValueError(f"{role} needs a named rule.")
        if not str(rule.get("text", "")).strip():
            raise ValueError(f"{role} needs rule text.")
    return policy


def departments_for_role(policy: dict[str, Any], role: str) -> set[str]:
    """Department folders this role may read. An unknown role reads none."""
    spec = policy["roles"].get(role)
    if not isinstance(spec, dict):
        return set()
    return set(spec["departments"])


def rule_for_role(policy: dict[str, Any], role: str) -> dict[str, str] | None:
    """The named rule on this role, or None when the role is not listed."""
    spec = policy["roles"].get(role)
    if not isinstance(spec, dict):
        return None
    rule = spec["rule"]
    return {"id": rule["id"], "text": rule["text"]}


def rule_for_department(policy: dict[str, Any], department: str) -> dict[str, str]:
    """The rule that names who may read this folder.

    A role whose department list is exactly this folder owns the rule.
    A role that includes the folder among others is the fallback.
    """
    fallback = None
    for spec in policy["roles"].values():
        departments = list(spec["departments"])
        rule = {"id": spec["rule"]["id"], "text": spec["rule"]["text"]}
        if departments == [department]:
            return rule
        if department in departments and fallback is None:
            fallback = rule
    if fallback is not None:
        return fallback
    return {
        "id": department,
        "text": f"No role in policy.yaml may read the {department} folder.",
    }


def refusal_text(role: str, department: str, rule: dict[str, str]) -> str:
    """Refusal that names the role, the folder, and the rule."""
    return (
        f"Your role ({role}) can't access {department} documents "
        f"(rule {rule['id']}). {rule['text']}"
    )


def departments_in(ranked: list[tuple]) -> list[str]:
    """Department folder names in ranked order, without duplicates."""
    found = []
    for item in ranked:
        department = item[3].get("department")
        if department and department not in found:
            found.append(department)
    return found


def sources_from_citations(citations: list[dict[str, Any]]) -> list[str]:
    sources = []
    for citation in citations:
        label = (
            f"{citation.get('source_file') or 'unknown'} p. {citation.get('page_number')}"
        )
        if label not in sources:
            sources.append(label)
    return sources


def trail_for(role, rule, departments, sources) -> dict[str, Any]:
    copied = None if rule is None else {"id": rule["id"], "text": rule["text"]}
    return {
        "role": role,
        "rule": copied,
        "departments": list(departments),
        "sources": list(sources),
    }


def outcome(answer, citations, trail, stage) -> dict[str, Any]:
    return {
        "answer": answer,
        "citations": citations,
        "trail": trail,
        "stage": stage,
    }


def get_reranker():
    """Load the engine's CPU cross-encoder once."""
    global _RERANKER
    if _RERANKER is not None:
        return _RERANKER
    try:
        _RERANKER = engine.load_reranker()
    except ImportError as exc:
        raise SystemExit(
            "sentence-transformers is not installed. "
            "Run: pip install -r requirements.txt"
        ) from exc
    except Exception as exc:
        raise RuntimeError(
            f"Reranker model '{engine.RERANKER_MODEL}' is not available locally."
        ) from exc
    return _RERANKER


def surviving_passages(reranked: list[tuple]) -> list[tuple]:
    """Reranked passages at or above the cutoff, capped at the context count.

    The list is already highest-first. The cutoff itself is unchanged: a top
    score below -0.7 never reaches this function.
    """
    kept = []
    for item in reranked:
        if engine.rerank_below_cutoff(item[0]):
            break
        kept.append(item)
        if len(kept) >= engine.FINAL_CONTEXT_COUNT:
            break
    return kept


def consult(question: str, role: str) -> dict[str, Any]:
    """Retrieve across every department, then allow or refuse.

    Qwen is called once, and only when a surviving passage is from a
    department this role may read. A refusal names the rule and does not
    call Qwen. A top rerank score below -0.7 returns "I don't know." and
    does not call Qwen either.
    """
    question = question.strip()
    role = role.strip()
    policy = load_policy()
    if not CORPUS_DIR.is_dir():
        raise SystemExit(f"Missing PDF folder: {CORPUS_DIR}")
    pdfs = engine.list_pdfs(CORPUS_DIR)
    if not pdfs:
        raise SystemExit(f"No PDFs found in {CORPUS_DIR}.")

    index = engine.load_corpus(pdfs)
    candidates = engine.rank_candidates(question, index)
    reranked = engine.rerank_candidates(question, candidates, get_reranker())
    top_score = reranked[0][0] if reranked else None
    if not reranked or engine.rerank_below_cutoff(top_score):
        return outcome(
            "I don't know.",
            [],
            trail_for(role, None, [], []),
            "cutoff",
        )

    surviving = surviving_passages(reranked)
    allowed_names = departments_for_role(policy, role)
    allowed = [
        item for item in surviving if item[3].get("department") in allowed_names
    ]
    if not allowed:
        blocked_departments = departments_in(surviving)
        rules = [
            (department, rule_for_department(policy, department))
            for department in blocked_departments
        ]
        answer = " ".join(
            refusal_text(role, department, rule) for department, rule in rules
        )
        rule = rules[0][1] if rules else None
        return outcome(
            answer,
            [],
            trail_for(role, rule, blocked_departments, []),
            "refused",
        )

    generated = curator.generate_answer(
        question, allowed, "policy", reranked=True, quiet=True
    )
    citations = generated["citations"]
    return outcome(
        generated["answer"],
        citations,
        trail_for(
            role,
            rule_for_role(policy, role),
            departments_in(allowed),
            sources_from_citations(citations),
        ),
        "answered",
    )


def print_result(result: dict[str, Any]) -> None:
    """Print the answer, the citations, and the trail."""
    print(f"Answer: {result['answer']}")
    if result["stage"] == "refused":
        print("Answer call skipped.")
    elif result["stage"] == "cutoff":
        print(
            f"Top rerank score is below {engine.RERANK_ABSTAIN_BELOW}. "
            "Answer call skipped."
        )
    for citation in result["citations"]:
        source = citation.get("source_file") or "unknown"
        print(f"Source: {source}, page {citation.get('page_number')}")
    trail = result["trail"]
    rule = trail["rule"]
    print("\n--- Trail ---")
    print(f"Role: {trail['role']}")
    if rule:
        print(f"Rule: {rule['id']} — {rule['text']}")
    else:
        print("Rule: none")
    departments = ", ".join(trail["departments"]) if trail["departments"] else "none"
    sources = ", ".join(trail["sources"]) if trail["sources"] else "none"
    print(f"Departments: {departments}")
    print(f"Sources: {sources}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Answer from the department folders this role may read."
    )
    parser.add_argument("--role", required=True, help="Role from policy.yaml.")
    parser.add_argument("--question", required=True, help="Question to answer.")
    args = parser.parse_args(argv)
    print_result(consult(args.question, args.role))
    return 0


if __name__ == "__main__":
    sys.exit(main())
