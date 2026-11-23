# 🧬 MSigDB Signature Generator

**▶️ Live app: [sig-gen.streamlit.app](https://sig-gen.streamlit.app)**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.62-FF4B4B.svg)](https://streamlit.io/)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.19386462-blue.svg)](https://doi.org/10.5281/zenodo.19386462)

Turn a plain-language biological question into a set of distinct gene signatures, each one traceable to the gene sets and statistics that produced it.

> **The LLM proposes, the knowledge base decides.** A language model breaks the question into mechanisms and suggests seed genes. Which genes end up in a signature is decided by enrichment statistics over ~34,000 MSigDB gene sets, not by the model.

---

## What it does

| Layer | What happens | Needs |
|---|---|---|
| **1. Mechanisms** | An LLM decomposes the question into N distinct mechanisms, each with a plain-language description and seed genes. Aliases (GLUT1, PD-L1, 4-1BB) are mapped to official symbols through NCBI Gene. | any LLM provider; NCBI (optional) |
| **2. Signatures** | Gene sets enriched for the seeds (hypergeometric, BH-FDR) are retrieved, re-ranked by semantic similarity to the description, and genes are scored by weighted support across them. Deterministic, runs locally. | — |
| **3. Network expansion** | Optionally adds genes that are network neighbours of at least two signature genes. | pathway API |
| **4. Verification** | Either one model reviews each signature, or three models with opposed burdens of proof **debate every gene** using per-gene database evidence. | any LLM provider; pathway API (optional) |
| **5. Approve & export** | You have the last word on every gene. Export GMT, CSV and a JSON record with full provenance. | — |

### Does the retrieval work?

Held-out recovery benchmark: hide a curated gene set (and its near-duplicates) from the knowledge base, give the method its name plus a few member genes, and score the top 20 other genes it returns against the hidden members. Test split, 250 gene sets never used for tuning ([protocol and full table](docs/METHODS.md#4-validation)):

| Seeds given | v2 pipeline | semantic only | enrichment only | **this method** |
|---|---|---|---|---|
| 10, all correct | 0.04 | 0.35 | 0.68 | **0.70** |
| 10, 30% wrong | 0.03 | 0.34 | 0.61 | **0.64** |
| 10, 50% wrong | 0.02 | 0.33 | 0.55 | **0.60** |
| 5, all correct | 0.03 | 0.36 | 0.65 | **0.69** |

Precision@20; random genes score 0.001. Reproduce with `python benchmarks/heldout_recovery.py`.

### The debate

Three models vote keep/remove **on every gene**, each with a different burden of proof (skeptic, advocate, mediator). They see the research question, the mechanism, the genes, and a database evidence table (links to the rest of the signature, GTEx expression, knowledge-base coverage). Only contested genes are re-argued. A gene is voted out when the confidence-weighted pooled probability of removal is at least 0.6 *and* a majority voted to remove it. Every signature gets per-gene entropy and conflict, a stability score, agreement, Fleiss' κ, and a count of how often the consensus followed or overrode the database.

In the app, voted-out genes are deselected with the reason shown — never silently deleted — so you can overrule them.

---

## Quick start

```bash
git clone https://github.com/arunviswanathan91/msigdb-signature-generator.git
cd msigdb-signature-generator
pip install -r requirements.txt
streamlit run app.py
```

In the sidebar open **Providers & model**, pick a provider, paste its API key and click **Connect**. You can connect several at once.

### Without a browser

The command-line runner executes the same code as the app, so a result can be regenerated from one line:

```bash
export GROQ_API_KEY=...        # and/or OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY, DEEPSEEK_API_KEY, ...
python run_pipeline.py "T cell exhaustion in the tumor microenvironment" \
    --mechanisms 8 --tissue "Blood/PBMC" --disease melanoma \
    --expand --debate --out results/exhaustion
```

```bash
python run_pipeline.py --check     # which providers, models, debate roles and services are available
```

---

## LLM providers

Any of these work, alone or together. Models are discovered from each provider at run time; nothing is hardcoded.

| Provider | Environment variable | Notes |
|---|---|---|
| Groq | `GROQ_API_KEY` | free tier, with tight per-minute and per-day token limits |
| OpenAI | `OPENAI_API_KEY` | |
| Anthropic (Claude) | `ANTHROPIC_API_KEY` | through the official `anthropic` SDK |
| Google Gemini | `GEMINI_API_KEY` | |
| Mistral AI | `MISTRAL_API_KEY` | |
| DeepSeek | `DEEPSEEK_API_KEY` | |
| xAI (Grok) | `XAI_API_KEY` | |
| OpenRouter | `OPENROUTER_API_KEY` | one key for many vendors |
| Together AI | `TOGETHER_API_KEY` | |
| Fireworks AI | `FIREWORKS_API_KEY` | |
| Cerebras | `CEREBRAS_API_KEY` | |
| Cohere | `COHERE_API_KEY` | |
| Perplexity (Sonar) | `PERPLEXITY_API_KEY` | no model list; type the model ID |
| Hugging Face Inference Providers | `HF_TOKEN` | |
| NVIDIA NIM | `NVIDIA_API_KEY` | |
| Ollama (local) | — | only when the app runs on the same machine |
| Custom | — | any OpenAI-compatible endpoint: Azure OpenAI v1, LM Studio, vLLM, LiteLLM |

A model is written `provider:model`, for example `anthropic:claude-opus-5-5` or `groq:openai/gpt-oss-120b`.

**Why connect more than one?** The debate is only as independent as its three models. With keys for different vendors, the three roles are automatically drawn from different model families. With a single provider the app tells you how many distinct families it could find.

**What the client handles for you.** A parameter a model rejects is dropped and the call retried; truncated answers are retried with a larger budget and never parsed; per-minute limits are waited out; a model that is retired, out of quota or declines the request hands over to another connected model. Every substitution is shown and exported.

**Only Groq has been exercised against the live service.** The other providers are implemented from their documentation and covered by tests with faked transports. Model IDs are whatever each provider's own model list returns.

Keys entered in the app stay in the browser session and are never written to exports. A key set as an environment variable (or Streamlit secret) connects that provider automatically for every visitor of that deployment.

---

## Architecture

```text
                    ┌──────────────────────────┐
   your browser ───▶│  Streamlit app (app.py)  │     run_pipeline.py (same code, no UI)
                    └───┬──────────┬─────────┬─┘
                        │          │         │
            LLM calls   │          │         │  gene aliases
                        ▼          │         ▼
          ┌────────────────────┐   │   ┌──────────────────┐
          │ any LLM provider   │   │   │ NCBI Gene        │
          │ (llm_client.py)    │   │   │ (E-utilities)    │
          └────────────────────┘   │   └──────────────────┘
                                   │  network + evidence
                                   ▼
                        ┌──────────────────────────┐      ┌──────────────────────────┐
                        │ HF Space: msigdb-api     │─────▶│ HF Dataset:              │
                        │ (REST)                   │ loads│ msigdb-databases (~1.4GB)│
                        └──────────────────────────┘      └──────────────────────────┘

   Local, no network:  data/knowledge_base.json.gz   34,021 gene sets, 42,302 genes
                       data/pathway_embeddings.pkl   one embedding per gene set
```

If the Space is asleep, Layers 1, 2 and 5 still work, Layer 3 is disabled, and the debate runs without database evidence and says so in the export. If NCBI is unreachable, aliases are simply not resolved.

### Waking the pathway API

Free Hugging Face Spaces sleep when unused. Open [the Space page](https://huggingface.co/spaces/arunviswanathan91/msigdb-api), leave it open for 30–60 seconds until it reads *Running*, then click **Retry backend** in the app. If it shows a runtime error instead, the owner needs **Settings → Factory rebuild**.

---

## Repository

| File | Role |
|---|---|
| `app.py` | Streamlit UI only |
| `run_pipeline.py` | headless runner and `--check` diagnostics |
| `signature_engine.py` | retrieval, scoring, sub-modules (pure NumPy/SciPy, deterministic) |
| `pipeline.py` | steps shared by app and runner: decomposition, alias resolution, review, applying outcomes |
| `debate_engine.py` | structured per-gene debate and its statistics |
| `llm_client.py` | JSON chat calls with recovery; OpenAI-compatible and Anthropic backends |
| `providers.py` | the provider table |
| `model_registry.py` | model references, ranking, debate role assignment |
| `db_client.py` | pathway API: network expansion, per-gene evidence |
| `ncbi_client.py` | NCBI Gene alias lookup |
| `exporters.py` | GMT, CSV, provenanced JSON |
| `gene_filters.py` | housekeeping-gene filter |
| `kb_builder.py`, `scripts/build_embeddings.py` | rebuild the knowledge base and its embeddings |
| `benchmarks/heldout_recovery.py` | the validation benchmark; results in `benchmarks/results/` |
| `tests/` | 180 offline tests, including the app driven end to end |
| `docs/METHODS.md` | the method, formulas, validation and limitations |

```bash
pip install -r requirements-dev.txt
pytest                                           # ~30 s, no network
python scripts/build_embeddings.py --verify      # the committed embeddings match the knowledge base
```

---

## Reproducibility

Every JSON export records: the query and context; the knowledge-base version and SHA-256 of both data files; every retrieval parameter; the mechanisms, seed genes and alias mappings; for each signature the supporting gene sets with p-values and each gene's score, support and origin; the provider and model that actually answered each call, with the sampling controls that were really applied; every debate vote with its reason; every gene left out, with why; and package versions.

Retrieval and scoring are bit-for-bit deterministic. LLM steps use temperature 0 and a fixed seed where the provider accepts them. Some models do not (current Claude models and OpenAI reasoning models fix their own sampling); the export shows `temperature: null` for those calls, and their output can differ between runs.

---

## Limitations

- **The benchmark measures recovery of curated gene-set members**, with seeds drawn from the hidden set. It shows the retrieval finds related genes and tolerates wrong seeds. It does not show that a signature is biologically correct for a new question; that needs external validation on expression data.
- **Semantic search is the weak channel.** Gene-set embeddings encode the set name plus its first ten genes; on its own it scores half of what enrichment does. It earns its place by re-ranking and by backing up noisy seeds.
- **Debate outcomes depend on the models.** Three models from one provider share blind spots. The membership rule the agents apply is strict by design; a signature that is heavily pruned is flagged, not hidden.
- **Database evidence is partial.** The evidence table covers 10,000 genes, the network stores each gene's 50 strongest links, and GTEx is bulk tissue, which can miss genes restricted to rare cell types. How `gene_gene.db` and the evidence table were built is not documented in this repository and must be described in any paper.
- **Free-tier limits.** A 35-signature debate needs roughly 150k tokens per model, close to a free daily allowance. The client waits and substitutes models, but large runs are slow on free keys.
- **Licensing of the data.** `data/knowledge_base.json.gz` and the Hugging Face dataset redistribute MSigDB content, including KEGG and BioCarta gene sets that carry their own terms. Check the [MSigDB licence](https://www.gsea-msigdb.org/gsea/msigdb_license_terms.jsp) before publishing, and give the dataset a data card. No `LICENSE` file is present for the code yet.
- **The Space still exposes unauthenticated cache endpoints** (`/api/cache/*`). The app no longer uses them; they should be removed from the Space.

---

## Citation

```bibtex
@software{msigdb_signature_generator,
  author  = {Viswanathan, Arun Geetha},
  title   = {MSigDB Signature Generator},
  year    = {2026},
  doi     = {10.5281/zenodo.19386462},
  url     = {https://github.com/arunviswanathan91/msigdb-signature-generator}
}
```

## Acknowledgments

- **MSigDB** (Broad Institute) and **ImmPort** — gene sets
- **NCBI Gene** — gene nomenclature; the idea of grounding gene names in NCBI's APIs follows **GeneGPT** (Jin *et al.*, *Bioinformatics* 2024)
- **GTEx** — tissue expression
- **sentence-transformers** — `all-MiniLM-L6-v2` embeddings
- **Streamlit** — application framework
