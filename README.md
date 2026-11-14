# Policy-first RAG

A question and a user role are checked against `policy.yaml`. One short call to local `qwen2.5:3b` lists every department the question needs, each with a reason. Python checks the role against each department. A department the role cannot read is refused with that access rule, and its files stay closed.

Allowed folders are the only ones opened. Chunking and `nomic-embed-text` follow `curator.py`. Cosine similarity keeps the top 10 chunks. The cross-encoder reranker reorders those 10, and both orders are printed side by side. One `qwen2.5:3b` call answers from the top 6 reranked passages and cites the page.

Tax, Legal, and HR each have a folder, allowed roles, and a few rules. The sample firm, Aster Meridian Group, is fictional.

## Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
ollama pull nomic-embed-text
ollama pull qwen2.5:3b
python3 policy_rag.py
```

One question:

```bash
python3 policy_rag.py --role tax_analyst --question "When is the second quarterly estimated tax voucher due?"
```

The five demo questions are in `demo_questions.json`:

- `routed` — a tax analyst gets the estimated-tax date, with a page citation.
- `routed-hr` — an HR partner gets the parental-leave length.
- `cross-dept` — counsel asks a legal question and a tax question. Legal is opened. Tax is refused under `TAX-01`.
- `refusal` — a tax analyst asks a legal question and is refused under `LEG-01`.
- `unknown` — counsel asks about litigation holds, which the files do not state, so the answer is "I don't know."

Each answer ends with a trail: the role, the rules applied, the departments chosen and why, and the sources cited. An allowed question also prints the cosine order beside the reranked order.

Regenerate the six PDFs with `python3 scripts/build_docs.py`. Checks that do not call Qwen:

```bash
python3 -m unittest discover -s tests -t . -v
```

## Cached folder search on an M1 Air (8 GB)

`curator_cached.py` indexes every PDF in a folder, one cache file per PDF. It keeps cosine similarity to the top 10 passages, reranks those 10 with the CPU cross-encoder, and makes one `qwen2.5:3b` call on the top 6. Each answer prints search, rerank, and answer times, and cites the file and page. `qwen2.5:3b` and `nomic-embed-text` are the models that fit this machine. The reranker stays on CPU (`cross-encoder/ms-marco-MiniLM-L6-v2`).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
ollama pull nomic-embed-text
ollama pull qwen2.5:3b
python3 curator_cached.py --folder knowledge_test --question "What is the standard vacation entitlement for a full year?"
python3 eval/run_eval.py
```

`--folder` defaults to `knowledge_test/` (HR, Legal, and Tax). Pass `--pdf` to index one file instead. `eval/run_eval.py` reads `eval/questions.yaml` and writes `eval/results.csv` plus `eval/summary.txt`.
