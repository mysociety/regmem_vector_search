
# regmem_vector_search

Rolling vector search of UK parliamentary debates for declarations of interest, published via [GitHub Pages](https://pages.mysociety.org/regmem_vector_search/).

## What this does

Parliamentary rules require MPs to declare relevant financial interests when speaking in debates. This project uses vector similarity search to find speeches that mention the Register of Members' Financial Interests, then uses an LLM agent to evaluate whether the declaration meets the clarity requirements.

The pipeline has three stages:

1. **Vector search** — [`mini-transcript-search`](https://github.com/mysociety/mini-transcript-search) downloads XML transcripts from TheyWorkForYou, computes sentence embeddings, and finds speeches similar to a set of reference phrases about declaring interests.
2. **Agent evaluation** — Speeches flagged by the vector search are passed to an OpenAI GPT-4o agent (`agent_refine.py`) that determines whether each speech actually contains a declaration, and whether the declaration clearly states the nature of the interest (as opposed to a vague reference to the register).
3. **Publishing** — Results are rendered as a Jekyll site and deployed to GitHub Pages, with a daily-updated list of possible mentions from the last 30 days.

## Project structure

```
src/regmem_vector_search/
├── search.py          # Vector search logic, result caching, text processing
├── agent_refine.py    # LLM agent for evaluating declaration clarity
├── config.py          # Settings (API keys via .env)
└── __main__.py        # CLI entry point

notebooks/
├── infer_last_month.ipynb   # Daily report: last 30 days of declarations
├── infer_last_year.ipynb    # One-off: bulk search over a year
└── split_last_year.ipynb    # Splits yearly results into per-MP pages

docs/                  # Jekyll site published to GitHub Pages
```

## Caching

Three layers of caching avoid redundant API calls and computation:

- **Transcript embeddings** — Per-day parquet files stored in `data/parlparse_xmls/`. Once a day's transcripts are embedded, they are reused on subsequent runs.
- **Search results** — The full `SearchResult` from each query is cached in `data/regmem_vector_search/search_results.sqlite` using `PydanticDBM`, keyed by date range and threshold. Same-day re-runs skip loading parquets and computing cosine similarity.
- **Agent declarations** — Per-speech LLM evaluations are cached in `data/regmem_vector_search/interest_declarations.sqlite`. Already-evaluated speeches are not re-sent to the API.

In GitHub Actions, `data/parlparse_xmls/` and `data/regmem_vector_search/` are preserved across runs using `actions/cache`.

## Setup

This project uses a devcontainer. To run locally:

1. Clone with submodules: `git clone --recurse-submodules`
2. Create a `.env` file with:
   ```
   OPENAI_APIKEY=sk-...
   HF_TOKEN=hf_...
   ```
3. Open in VS Code with the Dev Containers extension, or build with `docker compose build`.

## Running

```bash
# Run the daily search notebook
notebook render search --publish

# Run tests
script/test

# Lint
script/lint
```

## GitHub Actions

The `build_and_publish` workflow runs daily at 08:00 UTC. It:

1. Restores cached data from prior runs
2. Runs `infer_last_month.ipynb` inside the devcontainer
3. Commits any updated output to the repo
4. Builds the Jekyll site and deploys to GitHub Pages
5. Sends a Slack notification on success or failure
