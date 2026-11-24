"""Friday demo. Ask any question as a role from policy.yaml.

One shot:

    python3 demo.py --role tax_analyst --question "When is the voucher due?"

Leave off --role or --question and it prompts for a role, then keeps asking
until the question line is blank. ask() is what a Thursday UI calls.
"""

import argparse
import sys
from pathlib import Path

import policy_rag


def ask(question: str, role: str) -> dict:
    """Answer one question for one role.

    The role has to be listed in policy.yaml. The return value has the
    answer, file and page citations, and the trail (role, rules, departments
    and why, sources). No question is built in.
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Enter a question.")
    if not isinstance(role, str) or not role.strip():
        raise ValueError("Enter a role.")
    policy = policy_rag.load_policy()
    cleaned_role = role.strip()
    if cleaned_role not in policy_rag.known_roles(policy):
        raise ValueError(policy_rag.unknown_role_message(cleaned_role, policy))
    outcome = policy_rag.consult(policy, cleaned_role, question.strip())
    citations = []
    for citation in outcome["citations"]:
        source = citation.get("source") or ""
        citations.append(
            {
                "file": Path(source).name if source else "",
                "page": citation.get("page_number"),
                "quote": citation.get("quote", ""),
                "source": source,
            }
        )
    parts = policy_rag.trail_parts(
        cleaned_role, outcome["decisions"], outcome["citations"]
    )
    sources = []
    for citation in citations:
        if not citation["file"] or citation["page"] is None:
            continue
        label = f"{citation['file']}, page {citation['page']}"
        if label not in sources:
            sources.append(label)
    return {
        "question": question.strip(),
        "role": cleaned_role,
        "answer": outcome["answer"],
        "citations": citations,
        "trail": {
            "role": cleaned_role,
            "rules": parts["rules"],
            "departments": parts["departments"],
            "sources": sources,
        },
    }


def print_result(result: dict) -> None:
    """Print the answer, file and page citations, and the trail."""
    print(f"Answer: {result['answer']}")
    print()
    if result["citations"]:
        seen = []
        for citation in result["citations"]:
            label = f"{citation['file']}, page {citation['page']}"
            if label in seen:
                continue
            seen.append(label)
            print(f"Citation: {label}")
    else:
        print("Citation: none")
    trail = result["trail"]
    print()
    print("Trail")
    print(f"Role: {trail['role']}")
    print("Rules: " + ("; ".join(trail["rules"]) if trail["rules"] else "none"))
    departments = trail["departments"]
    print("Departments: " + ("; ".join(departments) if departments else "none"))
    print("Sources: " + ("; ".join(trail["sources"]) if trail["sources"] else "none"))


def _prompt_role(roles: list[str]) -> str | None:
    print("Roles: " + ", ".join(roles))
    while True:
        try:
            entered = input("Role: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return None
        if not entered:
            return None
        if entered in roles:
            return entered
        print(policy_rag.unknown_role_message(entered))


def _interactive(preset_role: str | None, preset_question: str | None) -> int:
    policy = policy_rag.load_policy()
    roles = policy_rag.known_roles(policy)
    role = (preset_role or "").strip()
    if role and role not in roles:
        print(policy_rag.unknown_role_message(role))
        role = ""
    if not role:
        role = _prompt_role(roles)
        if role is None:
            return 0
    pending = preset_question
    while True:
        if pending is not None:
            question = pending.strip()
            pending = None
            if not question:
                return 0
        else:
            try:
                question = input("Question: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return 0
            if not question:
                return 0
        print_result(ask(question, role))
        print()


def main(argv: list[str] | None = None) -> int:
    """Ask once when both flags are set. Otherwise prompt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question", default=None, help="Question to ask.")
    parser.add_argument(
        "--role",
        default=None,
        help="Role from policy.yaml. Leave this off to be prompted.",
    )
    args = parser.parse_args(argv)
    if args.role and args.question:
        try:
            result = ask(args.question, args.role)
        except ValueError as exc:
            parser.error(str(exc))
        print_result(result)
        return 0
    return _interactive(args.role, args.question)


if __name__ == "__main__":
    sys.exit(main())
