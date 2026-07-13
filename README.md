# Predictive Process Mining with Retrieval-Augmented LLMs

This repository contains the code and thesis material for a predictive process monitoring pipeline that combines:

- event log preprocessing from XES files
- semantic retrieval over historical prefixes using Qdrant
- embedding models from Sentence Transformers
- optional reranking with BGE rerankers
- next-activity prediction with either OpenAI models or local Ollama models

The project is designed to support reproducible experiments on business process event logs such as BPI Challenge, Sepsis, and Hospital Billing datasets.

## Repository Structure

```text
predictive_process_mining_rag/
|-- data/
|   |-- xes_logs/                  # Raw XES logs and generated train/test splits
|   |-- evaluation_results/        # Stored evaluation outputs
|   `-- <generated_dataset_dirs>/  # retrieval.csv, test.csv, attribute_labels.json
|-- src/
|   |-- cli.py                     # Main CLI entrypoint
|   |-- event_log_preprocessing.py # XES parsing, splitting, prefix generation
|   |-- retrieval.py               # Qdrant storage and retrieval
|   |-- evaluation.py              # LLM-based evaluation and metrics
|   |-- llm_api.py                 # OpenAI/Ollama request layer
|   `-- utils.py                   # Reporting and helper utilities
|-- thesis/
|   `-- final_thesis.pdf           # Final thesis PDF for GitHub distribution
|-- requirements.txt
`-- README.md
```

## What the Pipeline Does

At a high level, the workflow is:

1. Read a raw `.xes` event log.
2. Split it into retrieval/train and test data.
3. Convert traces into prefix -> next-activity pairs.
4. Store prefix embeddings in a Qdrant vector database.
5. Retrieve the most similar historical prefixes for each test prefix.
6. Ask an LLM to predict the next activity.
7. Compute evaluation metrics such as accuracy, macro precision, macro recall, macro F1, and earlyness buckets.

## Requirements

- Python 3.11 recommended
- Docker, for running Qdrant locally
- Internet access for downloading embedding or reranker models from Hugging Face the first time
- Either:
  - an OpenAI API key for OpenAI-hosted models, or
  - a local Ollama installation for local inference

## Python Dependencies

The project uses the following main packages:

- `pandas`
- `pm4py`
- `sentence-transformers`
- `transformers`
- `qdrant-client`
- `python-dotenv`
- `openai`
- `rustxes`

Install everything with:

```bash
pip install -r requirements.txt
```

## Reproducible Setup

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd predictive_process_mining_rag
```

### 2. Create and activate a virtual environment

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

On Linux or macOS:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Start Qdrant

Run Qdrant locally with Docker:

```bash
docker run -d --name qdrant \
  -p 6333:6333 \
  -p 6334:6334 \
  -v "$(pwd)/qdrant_data:/qdrant/storage" \
  qdrant/qdrant
```

If you are using Windows PowerShell, this variant is usually safer:

```powershell
docker run -d --name qdrant `
  -p 6333:6333 `
  -p 6334:6334 `
  -v "${PWD}\qdrant_data:/qdrant/storage" `
  qdrant/qdrant
```

### 4. Configure environment variables

Create a `.env` file in the repository root.

Minimal configuration for Qdrant:

```env
QDRANT_GRPC_PORT=6334
OLLAMA_BASE_URL=http://localhost:11434
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_API_KEY=your_openai_api_key_here
```

Notes:

- `QDRANT_GRPC_PORT` is required by the retrieval pipeline.
- `OPENAI_API_KEY` is required when using `gpt-4o-mini`, `gpt-4.1-mini`, or `gpt-4.1-mini-2025-04-14`.
- `OLLAMA_BASE_URL` is required when using a local Ollama model instead of an OpenAI model.
- You do not need both OpenAI and Ollama for every run, but keeping both configured makes switching easier.

### 5. Place raw XES logs in the expected folder

Store source logs under:

```text
data/xes_logs/
```

Examples already referenced by the codebase include:

- `BPI_Challenge_2012.xes`
- `BPI_Challenge_2020_InternationalDeclarations.xes`
- `HospitalBilling.xes`
- `Sepsis.xes`

## Running the Pipeline

The main entrypoint is:

```bash
python src/cli.py
```

It exposes two subcommands:

- `preprocess`
- `eval`

## Step 1: Preprocess Event Logs

This step:

- reads XES logs
- generates prefixes and next labels
- creates `retrieval.csv` and `test.csv`
- optionally creates `attribute_labels.json`
- optionally stores embeddings in Qdrant

### Basic preprocessing example

```bash
python src/cli.py preprocess \
  --datasets data/xes_logs/BPI_Challenge_2012.xes \
  --base 1 \
  --gap 3 \
  --m 3 \
  --split-mode trace
```

### Preprocess and build vector collections

```bash
python src/cli.py preprocess \
  --datasets data/xes_logs/BPI_Challenge_2012.xes data/xes_logs/Sepsis.xes \
  --base 1 \
  --gap 3 \
  --m 3 \
  --split-mode trace \
  --trace-cross-dedup \
  --store-embeddings \
  --embedding-models bge minilm \
  --batch-size 32
```

### Important preprocessing arguments

- `--datasets`: one or more `.xes` file paths
- `--base`: first prefix extraction point
- `--gap`: stride between extracted prefixes
- `--m`: keep values from the last `m` events
- `--trace-identifier`: trace id column, default is `case:concept:name`
- `--test-set-proportion`: default is `0.3`
- `--split-mode`: `trace` or `row`
- `--trace-cross-dedup`: removes exact duplicates between retrieval and test for trace-level split
- `--store-embeddings`: pushes embeddings to Qdrant
- `--embedding-models`: `bge` and/or `minilm`
- `--batch-size`: embedding upsert batch size

### Generated outputs

Each preprocessing run creates a dataset directory under `data/`, for example:

```text
data/last_3_b1_g3_BPI_Challenge_2012/
data/last_3_b1_g1_Sepsis_trace_dedup/
data/b1_g3_HospitalBilling_row/
```

Typical contents:

```text
retrieval.csv
test.csv
attribute_labels.json
```

## Step 2: Run Evaluation

Evaluation loads the generated dataset folders, retrieves similar prefixes, calls an LLM, and stores metrics in JSON.

### Example with an OpenAI model

```bash
python src/cli.py eval \
  --datasets last_3_b1_g3_BPI_Challenge_2012 last_3_b1_g3_Sepsis \
  --embedding-models bge minilm \
  --top-k 3 \
  --max-rows 100 \
  --llm-model gpt-4o-mini \
  --prompt-variant prompt_1 \
  --run-label openai_baseline
```

### Example with reranking enabled

```bash
python src/cli.py eval \
  --datasets last_3_b1_g1_HospitalBilling_trace_dedup \
  --embedding-models bge \
  --top-k 3 \
  --reranker-model bge \
  --rerank-pool-k 10 \
  --max-rows 100 \
  --llm-model gpt-4.1-mini \
  --prompt-variant prompt_2 \
  --run-label reranked_eval
```

### Example with a local Ollama model

```bash
python src/cli.py eval \
  --datasets last_3_b1_g1_Sepsis_trace_dedup \
  --embedding-models minilm \
  --top-k 3 \
  --max-rows 50 \
  --llm-model llama3 \
  --prompt-variant prompt_1 \
  --run-label ollama_eval
```

### Important evaluation arguments

- `--datasets`: dataset folder names under `data/` or absolute/relative dataset paths
- `--embedding-models`: `bge` and/or `minilm`
- `--top-k`: number of retrieved examples passed to the LLM
- `--max-rows`: optional cap on evaluated rows
- `--llm-model`: required
- `--prompt-variant`: optional prompt configuration
- `--reranker-model`: currently supports `bge`
- `--rerank-pool-k`: candidate pool size before reranking, must be `>= top-k`
- `--run-label`: label used in saved reports
- `--results-path`: output JSON path, default is `data/evaluation_results/evaluation_runs.json`
- `--keep-individual-predictions`: stores row-level predictions
- `--earlyness-buckets`: bucket boundaries for early prediction analysis

## Supported Models

### Embedding models

- `bge` -> `BAAI/bge-small-en-v1.5`
- `minilm` -> `sentence-transformers/all-MiniLM-L12-v2`

### Reranker models

- `bge` -> `BAAI/bge-reranker-base`

### LLM backends

OpenAI-routed models currently recognized by the code:

- `gpt-4o-mini`
- `gpt-4.1-mini`
- `gpt-4.1-mini-2025-04-14`

Any other model name is treated as an Ollama model and sent to `OLLAMA_BASE_URL`.

## Saved Results

Evaluation outputs are appended to:

```text
data/evaluation_results/evaluation_runs.json
```

Results are grouped by evaluation configuration, including:

- dataset
- base/gap/m preprocessing settings
- embedding model
- LLM model
- prompt variant
- reranker settings
- top-k and max-rows

## Reproducing a Full Experiment

For a clean end-to-end run:

1. Start Qdrant.
2. Create and activate the Python environment.
3. Add your `.env` file.
4. Place the raw XES logs inside `data/xes_logs/`.
5. Run `preprocess` with `--store-embeddings`.
6. Run `eval` on the generated dataset directory names.
7. Inspect `data/evaluation_results/evaluation_runs.json`.

Example:

```bash
python src/cli.py preprocess \
  --datasets data/xes_logs/BPI_Challenge_2012.xes data/xes_logs/Sepsis.xes \
  --base 1 \
  --gap 3 \
  --m 3 \
  --split-mode trace \
  --trace-cross-dedup \
  --store-embeddings \
  --embedding-models bge minilm

python src/cli.py eval \
  --datasets last_3_b1_g3_BPI_Challenge_2012 last_3_b1_g3_Sepsis \
  --embedding-models bge minilm \
  --top-k 3 \
  --max-rows 100 \
  --llm-model gpt-4o-mini \
  --prompt-variant prompt_1 \
  --run-label reproduction_run
```

## Thesis

The final thesis PDF intended for GitHub distribution lives at:

```text
thesis/final_thesis.pdf
```

The LaTeX working sources are kept out of version control and are not part of the public repository snapshot.

## Troubleshooting

### Qdrant connection errors

Check that:

- Docker is running
- the Qdrant container is up
- port `6334` is exposed
- `QDRANT_GRPC_PORT` matches the running container

### Missing OpenAI credentials

If you use an OpenAI model, confirm:

- `OPENAI_API_KEY` is set in `.env`
- `OPENAI_BASE_URL` is correct

### Ollama request failures

If you use a local model, confirm:

- Ollama is running
- `OLLAMA_BASE_URL` points to the local server
- the selected model is already pulled locally

### Slow first run

The first run may be slower because:

- sentence-transformer models are downloaded
- reranker weights are downloaded
- embeddings are created and uploaded to Qdrant

## Citation

If you use this repository in academic work, please cite the associated thesis and reference this GitHub repository.

## License

Add your preferred license here, for example `MIT`, `Apache-2.0`, or a thesis-specific academic usage license.
