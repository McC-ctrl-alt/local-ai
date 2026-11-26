# Policy-checked RAG

A question is retrieved across every department folder under `knowledge_test/`. Chunking, `nomic-embed-text`, cosine top 10, the CPU cross-encoder, and the -0.7 cutoff live in `engine.py`. `curator.py` and `policy_rag.py` both import it. The words in the question do not choose a folder.

`policy.yaml` names four roles. `hr_partner`, `legal_counsel`, and `tax_analyst` may each read their own department. `managing_partner` may read all three. Each role has one named rule.

Retrieval runs first. Cosine keeps the top 10 passages, the cross-encoder reranks those 10, and a top score below -0.7 returns "I don't know." without calling Qwen. If every passage that survives that cutoff is from a department the role cannot read, the answer names the rule and does not call Qwen. Otherwise one `qwen2.5:3b` call answers from the allowed passages only and cites the page. The call uses temperature 0, seed 42, and `num_ctx` 8192. `CURATOR_LLM` overrides the model (default `qwen2.5:3b`).

A new PDF dropped into `knowledge_test/hr`, `knowledge_test/legal`, or `knowledge_test/tax` is picked up on the next run. The embedding cache refreshes when a file changes. The firm, Northstar Advisory Group, is fictional.

## Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
ollama pull nomic-embed-text
ollama pull qwen2.5:3b
python3 policy_rag.py --role hr_partner --question "During a new hire's opening week on the job, what does their manager need to confirm?"
```

Ollama must be installed separately (`ollama pull nomic-embed-text` and `ollama pull qwen2.5:3b`).

The same question as `tax_analyst` is refused under `HR-01`. Qwen is not called.

Each answer ends with a trail: the role, the rule, the departments, and the sources cited.

Checks that do not call Qwen:

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
python3 curator.py --pdf knowledge_test/hr/HR_Department.pdf --question "What is the standard vacation entitlement for a full year?"
```

`curator.py --folder` defaults to `knowledge_test/`. HR, Legal, and Tax are the `hr/`, `legal/`, and `tax/` subfolders, and a PDF in any subfolder is indexed. Pass `--pdf` to index one file instead. `eval/run_eval.py` reads `eval/questions.yaml` and writes `eval/results.csv` plus `eval/summary.txt`.

Engine PRs must pass `python3 eval/compare.py` before merge. It compares a results CSV to the baseline `eval/results_baseline_mac.csv`. The report shows those counts, the per-question rerank scores, and the average search, rerank, and answer times. The command exits non-zero if any count gets worse, or if any no-answer question gets an answer. A rerank score that moves by more than 0.01 is marked CHANGED on that question's line.
