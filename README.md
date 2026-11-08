# Local document assistant

This demo answers a question about a folder of PDFs on your own computer.
It uses [Ollama](https://ollama.com) to run Qwen locally. It does not send the
documents to a paid API, and it does not need an API key.

The files in `sample_documents/` are fiction. Northwind Advisory LLC, Harborline
Markets, and Cedar & Pine Retail Cooperative are not real. Each PDF says so on
its first page.

## One-time setup

From the project folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Install Ollama from https://ollama.com/download, then:

```bash
ollama pull nomic-embed-text
ollama pull qwen2.5:3b
```

The first question also downloads a small public reranker,
`cross-encoder/ms-marco-MiniLM-L6-v2`. That download is free and does not need
an account. It runs on the CPU. The install is large because that reranker
uses PyTorch.

## Ask in the browser

```bash
streamlit run app.py
```

Open http://localhost:8501. Pick a sample question or type your own, then click
Ask. The page shows one answer, then the file name, page, and passage it used.

To ask about your own files, put PDFs in a folder and type that folder in the
sidebar. Only PDFs directly inside the folder are indexed. Click Reload
documents after you add or change a file.

## Ask in the terminal

```bash
python3 curator.py --question "What is the fixed fee in the Harborline Markets engagement letter?"
```

Leave off `--question` to type a question yourself. Another folder:

```bash
python3 curator.py --docs /path/to/your/pdfs --question "Your question"
```

## Sample questions the fictional files can answer

- What is the fixed fee in the Harborline Markets engagement letter?
- When is the Cedar & Pine inventory observation report due?
- How long does Northwind keep final engagement files?
- What late charge applies to unpaid invoices?
- Who must be told about a suspected confidentiality breach, and how soon?
- What personal investment amount must a Northwind staff member disclose?
- Does the Harborline letter authorize preparing a tax return?

This question is meant to be refused, because the files never state it:

- What was Harborline Markets' revenue in 2025?

## Checks that do not need the model

```bash
python3 -m unittest discover -s tests -t . -v
```

Those tests use stand-in replies. They do not prove that Qwen itself answered.
To try the real model, run one of the commands above while Ollama is open.

## What this demo does

The script reads every PDF in the folder, remembers which page each passage
came from, and asks Qwen once. Qwen must point at one of those passages. The
script then checks that the quoted words really appear on that file and page.
If the answer mentions a dollar amount, a number, or a time period, those words
must appear in the cited passage too. If the check fails, or the passages do
not contain the answer, you get `I don't know.` instead of a guessed answer.
The reply is kept short so a confused generation cannot run for many minutes.
