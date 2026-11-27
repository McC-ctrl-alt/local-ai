# Engine map

`eval/run_eval.py` scores `curator.py` on the ten questions in `eval/questions.yaml`. It does not call `policy_rag.py`. `curator.py` imports retrieval from `engine.py`. Search is cosine top 10, then the cross-encoder over those 10, then at most one `qwen2.5:3b` call. Nothing else retrieves passages.

## Stages

| Stage | File | What it does | Flags and settings |
| --- | --- | --- | --- |
| Choose the folder | `eval/run_eval.py` | Reads the question file and scores every PDF in the folder. | `--folder` defaults to `knowledge_test/`. `--questions` defaults to `eval/questions.yaml`. `--output` defaults to `eval/results.csv`. `--dump` writes `eval/dumps/<id>.txt`. |
| Chunk and cache | `engine.py` `build_page_aware_chunks`, `load_or_build_index`, `load_corpus` | Splits each page on `". "`, packs sentences into chunks of about 1000 characters with about 300 characters of overlap, and stores one cache file per PDF under `.curator_index/`. `list_pdfs` reads subfolders. Each chunk is tagged with `department`, the parent folder name. The cache still refreshes when the file's stat signature changes. | CLI: `--folder` defaults to `knowledge_test/`. `--pdf` indexes one file instead. `--index` only builds the cache. The eval calls `curator.load_corpus` on every PDF in its own `--folder`, including department subfolders. |
| Embed | `engine.py` `embed` | Embeds the question and each chunk with `nomic-embed-text` through Ollama. | `OLLAMA_URL` (default `http://localhost:11434`). Model name is `EMBED_MODEL`. |
| Cosine top 10 | `engine.py` `rank_candidates` | Keeps the 10 chunks with the highest cosine similarity. | `RAG_CANDIDATE_COUNT` (default `10`). |
| Cross-encoder rerank | `engine.py` `load_reranker`, `rerank_candidates` | Reorders those 10 on CPU. Loads ONNX from the local model folder. The eval logs the top score as `top_rerank_score`. | `RAG_RERANKER_MODEL` (default `cross-encoder/ms-marco-MiniLM-L6-v2`). |
| Abstain | `engine.py` `rerank_below_cutoff`, called from `curator.py` `ask_question` | If the top rerank score is below `-0.7`, the answer is `I don't know.` and Qwen is not called. On this folder the two no-answer questions scored `-3.37` and `-1.78`. The eight answerable questions scored from `0.32` to `7.17`. | Constant `RERANK_ABSTAIN_BELOW` (`-0.7`). Not a CLI flag. |
| One answer call | `curator.py` `generate_answer` via `engine.py` `chat_answer` | Sends the top 6 reranked passages to `qwen2.5:3b` (temperature 0, seed 42, num_ctx 8192, 128 tokens). The prompt asks for the general or standard rule, quoted from the sentence that states it, and `I don't know.` when no passage states the fact explicitly. At most two citations. The quote must appear on that page, or the answer becomes `I don't know.` | `RAG_FINAL_CONTEXT_COUNT` (default `6`). `CURATOR_LLM` overrides the model. CLI: `--question`, `--compare` (prints both orders and, when it does not abstain, answers twice). |
| Score | `eval/run_eval.py` | Labels each row `correct`, `partial`, `wrong`, or `correct-IDK`, and checks the cited file and page. | Same flags as the folder stage. |
| Gate | `eval/compare.py` | Compares a results CSV to the baseline. | Positional results path, default `eval/results.csv`. `--baseline` defaults to `eval/results_baseline_mac.csv`. Required for engine PRs. |

`curator.py` defaults to every PDF under `knowledge_test/`, including `hr/`, `legal/`, and `tax/`. `--pdf` answers one file instead. `eval/run_eval.py --folder` chooses the folder the eval scores. Dumps look up a file name inside that folder's subfolders.

Keyword rank in the CSV is a scorer-only count in `eval/run_eval.py`. It is not used to choose passages.

## Policy check after retrieval

`policy_rag.py` is not what the ten-question eval runs. It retrieves with `engine.py` across every department folder, then applies `policy.yaml`. Question wording does not choose a folder.

| Stage | File | What it does |
| --- | --- | --- |
| Retrieve | `engine.py` through `policy_rag.py` `consult` | Every PDF under `knowledge_test/`. Same defaults as curator: `RAG_CANDIDATE_COUNT` 10, `RAG_FINAL_CONTEXT_COUNT` 6, `RAG_RERANKER_MODEL`, `OLLAMA_URL`, `RERANK_ABSTAIN_BELOW` `-0.7`. The embedding cache is `.curator_index/`, one file per PDF, refreshed when the file's stat signature changes. |
| Allow or refuse | `policy_rag.py` `consult` | If every passage that survives the cutoff is from a department the role cannot read, the answer names that rule and Qwen is not called. Otherwise the one `chat_answer` call (temperature 0, seed 42, num_ctx 8192, `CURATOR_LLM`, default `qwen2.5:3b`) sees the allowed passages only. |
| Trail | `consult` return value | Role, rule, departments, and sources. `--role` and `--question` print the same trail. |

## What each eval question tests

The corpus is `knowledge_test/hr/HR_Department.pdf`, `knowledge_test/legal/Legal_Department.pdf`, and `knowledge_test/tax/Tax_Department.pdf` (Northstar Advisory Group, fictional).

| Id | Kind | Tests |
| --- | --- | --- |
| `hr-vacation-entitlement` | fact | HR page 1 says the standard entitlement is 15 working days per full year, and 20 days after five completed years. The answer has to be the standard year, not the five-year number. The right page is cosine rank 1. Top rerank score about `4.87`. |
| `hr-benefits-enrollment` | two-detail | One answer must include both facts on HR page 1: enrollment from the first of the month after 30 days, and the employer paying 75% of the standard health and dental premium. Top rerank score about `6.33`. |
| `hr-personnel-retention` | rephrased | The question says "ordinary personnel files" after someone has left. HR page 3 says routine personnel files are kept 7 years after employment ends. This is the weakest answerable score, about `0.32`, and it stays above the abstain line. |
| `tax-ontario-hst` | number+unit | Tax page 2 says a taxable service supplied in Ontario is generally subject to 13% HST. The label is `correct` only when the answer has both `13%` and `HST`. `13%` alone is `partial`. Top rerank score about `7.17`. |
| `legal-approval-band` | table-row | Legal page 1 has several fee bands. Total committed fees of CAD 100,001 to 500,000 require CFO + General Counsel + department VP, not the band below or above it. Top rerank score about `6.83`. |
| `legal-nda-term` | fact | Legal page 2 says the default NDA term is 3 years. Keyword rank is 2 and cosine rank is 1; rerank keeps the page at rank 1. Top rerank score about `3.48`. |
| `tax-return-due` | fact | Tax page 1 says a corporate income tax return is generally due six months after fiscal year-end. The same page also has an internal April 15 target. Cosine rank is 2 and rerank moves the page to rank 1. Top rerank score about `7.01`. |
| `hr-harassment-ack` | fact | HR page 2 says HR acknowledges harassment, discrimination, reprisal, violence, or an immediate safety concern within 1 business day. A nearby row uses 2 business days for pay and benefits. Top rerank score about `6.73`. |
| `hr-rrsp-match` | no-answer | No file states an employer RRSP match. The answer must be "I don't know." with no citation. Top rerank score about `-3.37`, so the abstain stage returns that without calling Qwen. |
| `hr-parental-weeks` | no-answer | HR page 1 says parental leave follows statute and the plan. It does not state a week count. The 75% health premium on that page is a nearby number. Top rerank score about `-1.78`, so the abstain stage returns "I don't know." without calling Qwen. |

`eval/results_run4.csv` is the baseline: 5 correct, 1 partial, 3 wrong, 1 correct-IDK, 8 of 10 citations. `python3 eval/compare.py` exits non-zero if a later run drops any of those counts, or if any no-answer question gets an answer. That last check does not depend on the baseline answer.
