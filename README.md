# Policy-first RAG

A question and a user role are checked against `policy.yaml`. Departments come from the words in that file, each with the policy sentence that matched. Python checks the role against each department. A department the role cannot read is refused with that access rule, and its files stay closed.

Allowed folders are the only ones opened. Chunking, `nomic-embed-text`, cosine top 10, the CPU cross-encoder, and the -0.7 cutoff live in `engine.py`. `curator.py` and `policy_rag.py` both import it. One `qwen2.5:3b` call answers from the top 6 reranked passages and cites the page. A top rerank score below -0.7 returns "I don't know." and does not call Qwen. The call uses temperature 0, seed 42, and `num_ctx` 8192. `CURATOR_LLM` overrides the model (default `qwen2.5:3b`).

Tax, Legal, and HR each have a folder, allowed roles, and a few rules. The sample firm, Aster Meridian Group, is fictional.

## Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
ollama pull nomic-embed-text
ollama pull qwen2.5:3b
python3 demo.py --role tax_analyst --question "When is the second quarterly estimated tax voucher due?"
```

`demo.ask(question, role)` is the function a UI calls. Leave off `--role` or `--question` and the script prompts for a role, then keeps asking until the question line is blank. The role has to be one of the roles in `policy.yaml`.

Start the local page with the venv active and Ollama running:

```bash
make demo
```

It binds to http://127.0.0.1:5000 on this machine only. The role dropdown is read from `policy.yaml`. Ask a question and the page shows the answer, file and page citations, and a "Why this answer" trail from `demo.ask()`. Startup sends one tiny Qwen call so the first question is not a cold load. The page waits up to 120 seconds and does not stream.

```bash
python3 policy_rag.py
```

Ollama must be installed separately (`ollama pull nomic-embed-text` and `ollama pull qwen2.5:3b`).

One question:

```bash
python3 policy_rag.py --role tax_analyst --question "When is the second quarterly estimated tax voucher due?"
```

The five demo questions are in `demo_questions.json`:

- `routed` — a tax analyst gets the estimated-tax date, with a page citation.
- `routed-hr` — an HR partner gets the parental-leave length.
- `cross-dept` — counsel asks a legal question and a tax question. Legal is opened. Tax is refused under `TAX-01`.
- `refusal` — a tax analyst asks a legal question and is refused under `LEG-01`.
- `unknown` — counsel asks about litigation holds. `policy.yaml` does not mention them, so no folder is opened and the answer is "I don't know."

Each `policy_rag.py` answer ends with a trail: the role, the rules applied, the departments chosen and why, and the sources cited. An allowed question also prints the cosine order beside the reranked order. `demo.py` prints the answer, the file and page, and that same trail.

Regenerate the six PDFs with `python3 scripts/build_docs.py`. Checks that do not call Qwen:

```bash
python3 -m unittest discover -s tests -t . -v
```

## Cached folder search on an M1 Air (8 GB)

`curator.py` indexes every PDF in a folder, one cache file per PDF. It keeps cosine similarity to the top 10 passages, reranks those 10 with the CPU cross-encoder, and makes one `qwen2.5:3b` call on the top 6. Each answer prints search, rerank, and answer times, and cites the file and page. `qwen2.5:3b` and `nomic-embed-text` are the models that fit this machine. The reranker stays on CPU (`cross-encoder/ms-marco-MiniLM-L6-v2`).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
ollama pull nomic-embed-text
ollama pull qwen2.5:3b
python3 eval/run_eval.py
python3 curator.py --pdf knowledge_test/HR_Department.pdf --question "What is the standard vacation entitlement for a full year?"
```

`curator.py --folder` defaults to `knowledge_test/` (HR, Legal, and Tax). Pass `--pdf` to index one file instead. `eval/run_eval.py` reads `eval/questions.yaml` and writes `eval/results.csv` plus `eval/summary.txt`.

Engine PRs must pass `python3 eval/compare.py` before merge. It compares a results CSV to the baseline `eval/results_baseline_mac.csv`. The report shows those counts, the per-question rerank scores, and the average search, rerank, and answer times. The command exits non-zero if any count gets worse, or if any no-answer question gets an answer. A rerank score that moves by more than 0.01 is marked CHANGED on that question's line.
