# 🧬 MSigDB Signature Generator

**▶️ Live app: [sig-gen.streamlit.app](https://sig-gen.streamlit.app)**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.62-FF4B4B.svg)](https://streamlit.io/)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.19386462-blue.svg)](https://doi.org/10.5281/zenodo.19386462)

Turn a plain-language biological question into a set of granular, non-overlapping gene signatures grounded in MSigDB pathways.

Ask *"T cell exhaustion in the tumor microenvironment"* and get back N distinct mechanisms, each backed by pathway evidence and refined by LLM verification — or by a multi-model adversarial debate.

---

## 🎯 What it does

The pipeline runs in five layers. Each is inspectable and re-runnable from the UI.

| Layer | What happens | Needs |
|---|---|---|
| **1. Granularity & Selection** | An LLM decomposes your query into exactly N non-overlapping molecular mechanisms. You pick which to keep. | Groq API |
| **2. Pathway Search** | Each mechanism is embedded and matched against ~19,000 MSigDB pathways by cosine similarity. Runs **locally**. | — |
| **3. DAM Expansion** | Expands signatures using gene-neighbour statistics from the pathway API. | Pathway API |
| **4. Verification** | Batch LLM review of each signature — flag genes to drop, suggest genes to add. Or run the **multi-LLM debate** instead. | Groq API |
| **5. Gene Addition & Export** | Confidence-scored additions, then export as GMT / CSV / JSON. | Pathway API (additions only) |

### The debate system

Instead of a single verification pass, three **different** models argue about each signature:

- **🔬 Skeptic** — distrusts LLM intuition, demands database evidence, hunts false positives
- **💡 Discoverer** — argues for plausible novel connections not yet in databases
- **⚖️ Mediator** — weighs both and breaks ties

A database injector feeds real evidence between rounds, and the final call is a confidence-weighted vote with convergence and entropy tracking. The three models are chosen live from your Groq account and are deliberately from **different model families** so the disagreement is genuine.

---

## 🏗️ Architecture

```text
                    ┌──────────────────────────┐
   your browser ───▶│  Streamlit app (app.py)  │
                    └────┬────────────────┬────┘
                         │                │
              chat/LLM   │                │  pathway lookups
                         ▼                ▼
              ┌──────────────────┐  ┌──────────────────────────┐
              │   Groq API       │  │  HF Space: msigdb-api    │
              │ (OpenAI-compat)  │  │  (REST backend)          │
              └──────────────────┘  └────────────┬─────────────┘
                                                 │ loads
                                                 ▼
                                    ┌──────────────────────────┐
                                    │ HF Dataset:              │
                                    │ msigdb-databases (~1.4GB)│
                                    └──────────────────────────┘

   Local, no network: data/pathway_embeddings.pkl  (Layer 2 search)
                      data/knowledge_base.json.gz  (19k pathways)
```

**The app never talks to the HF dataset directly** — only the Space does. If the Space is offline, Layers 1, 2 and 4 still work; Layers 3 and 5 and the debate system pause.

---

## 🚀 Quick Start

```bash
git clone https://github.com/arunviswanathan91/msigdb-signature-generator.git
cd msigdb-signature-generator
pip install -r requirements.txt
streamlit run app.py
```

Then in the sidebar:

1. Open **🔑 API & Models**
2. Paste a **Groq API key** — free at [console.groq.com/keys](https://console.groq.com/keys)
3. Click **Validate Key**

Validation and model discovery are the same call, so on success the sidebar immediately shows how many models your account can use and lets you pick which to run.

> **Note:** this app uses a **Groq** key, not a Hugging Face token. Earlier versions used HF inference; that migration is complete.

---

## 🤖 Model configuration

**No Groq model ID is hardcoded anywhere in the application code.** This is deliberate.

Groq retires models on a rolling schedule, and every ID this project ever pinned has since been decommissioned:

| Model | Retired |
|---|---|
| `llama-3.3-70b-versatile` | 2026-08-16 |
| `llama-3.1-8b-instant` | 2026-08-16 |
| `gemma2-9b-it` | 2025-10-08 |
| `llama-3.1-70b-versatile` | 2025-01-24 |
| `llama-3.2-90b-text-preview` | 2024-11-25 |

Hand-maintaining that list failed repeatedly — one "fix" swapped in two IDs that had already been dead for over a year, and the app 404'd on the next deploy.

So [`model_registry.py`](model_registry.py) never asserts which models exist. It **asks** Groq at runtime via `/models`, filters to chat-capable models, and picks the best available option from an ordered *preference* list. A retired ID in that list is harmless — it is simply skipped. If every preference is gone, the app falls back to whatever the API actually reports and keeps working.

**In the UI:** the sidebar exposes a **Reasoning model** and a **Fast model** picker, plus **↻ Refresh model list**. Layer 4 has its own per-run model dropdown.

**Programmatically:**

```python
from model_registry import fetch_available_models, pick_default, REASONING_PREFERENCE

models = fetch_available_models(api_key)          # live list
model  = pick_default(models, REASONING_PREFERENCE)
```

To change which models are *preferred*, edit the preference lists in `model_registry.py`. You never need to edit them just because a model was retired.

**If a model dies mid-run**, the app retries once on the next live model and tells you it did, rather than failing the layer.

---

## 📁 Repository structure

| File | Lines | Role |
|---|---|---|
| `app.py` | 3074 | Main Streamlit app — all five layers, sidebar, session state, exports |
| `debate_system_with_injector.py` | 1207 | Multi-round adversarial debate engine, weighted voting, convergence metrics |
| `model_registry.py` | 260 | **Model discovery and selection. Single source of truth for model IDs.** |
| `db_client.py` | 206 | REST client for the pathway API Space |
| `db_client_enhanced.py` | 553 | Builds debate evidence context from the same endpoints |
| `cache_client.py` | 218 | Semantic search cache (optional; degrades silently) |
| `kb_builder.py` | 205 | Builds a custom knowledge base from GMT uploads |
| `groq_model_diagnostic.py` | 230 | CLI: list live models, test debate roles |
| `test_debate_fixes.py` | 169 | CLI test of debate config and context growth |
| `test_cache_integration.py` | 120 | CLI test of the search cache |
| `complete_module_replacements.py` | 1111 | ⚠️ Legacy. Only `HOUSEKEEPING_GENES` is still used. |
| `material_ui_builtin.py` / `material_ui_css.py` / `material_design_theme.py` | 1168 | ⚠️ Theme modules, currently unreferenced by `app.py` |

### Data

| File | Size | Used by |
|---|---|---|
| `data/pathway_embeddings.pkl` | 52 MB | Layer 2 search. **Required** — the app hard-fails without it. |
| `data/knowledge_base.json.gz` | 8.8 MB | Built-in KB, ~19,000 pathways |
| `data/dam_index_light.pkl` | 39 MB | ⚠️ Committed but referenced by nothing. See Known gaps. |

### Config

- `.streamlit/config.toml` — disables the module watcher (see Troubleshooting)
- `requirements.txt` — fully pinned; see the file header for why

---

## 🌐 Backend services

| Service | Status | Impact if down |
|---|---|---|
| [Groq API](https://console.groq.com) | required | Layers 1 and 4 stop |
| [HF Space `msigdb-api`](https://huggingface.co/spaces/arunviswanathan91/msigdb-api) | optional-ish | Layers 3 and 5 and the debate pause; core generation is fine |
| [HF Dataset `msigdb-databases`](https://huggingface.co/datasets/arunviswanathan91/msigdb-databases) | indirect | Consumed by the Space, never by the app |

The sidebar shows a live **Pathway API** badge. When it is offline, the features that need it are disabled with an explanation instead of throwing, and a **↻ Retry backend** button re-probes.

### ☕ Waking the pathway API

Free Hugging Face Spaces **sleep after a period of inactivity**. This is normal and expected — nothing is broken.

**To wake it: open [the Space page](https://huggingface.co/spaces/arunviswanathan91/msigdb-api) and leave it open for ~30–60 seconds** until the status reads *Running*. Then return to the app and click **↻ Retry backend**.

Every offline message in the app links straight there, so users can do this themselves without knowing any of the above.

**If the Space shows a Runtime Error** instead of *Sleeping* (for example `Could not resolve host: huggingface.co` during container init — an HF infrastructure hiccup, not a code fault), the owner needs to open the Space → **Settings** → **Factory rebuild**.

---

## 🐛 Troubleshooting

**`Model unavailable` / 404 `model_not_found`**
Groq retired that model. Sidebar → **🔑 API & Models** → **↻ Refresh model list**, then pick another. The app also auto-retries once on its own.

**`Invalid API key`**
Check the key at [console.groq.com/keys](https://console.groq.com/keys). Note the sidebar now reports the *actual* failure — a timeout says timeout, not "bad key".

**`Pathway API: offline`**
The Space is down. Core generation still works. Factory-rebuild the Space to restore Layers 3/5 and the debate.

**Deploy log flooded with `ModuleNotFoundError: No module named 'torchvision'`**
Streamlit's module watcher walks `sys.modules` and touches `transformers.models.*.image_processing_*_fast`, each of which lazily imports `torchvision`. Harmless, but it buried real errors. Fixed by `fileWatcherType = "none"` in `.streamlit/config.toml`. For local hot-reload, set it back to `"auto"`.

**Knowledge base not found**
`data/knowledge_base.json.gz` must be present. The app checks `data/`, `./data/` and `../data/`.

---

## ⚠️ Known gaps

- `data/dam_index_light.pkl` (39 MB) is committed but referenced by no code. It looks intended as a local fallback for the gene-neighbour lookups that currently require the Space — wiring it up would remove that dependency entirely.
- `complete_module_replacements.py` is 1111 lines of legacy code kept alive for a single constant.
- The `material_ui_*` theme modules are unreferenced by `app.py`.
- `.gitignore` ignores `*.json` globally, so new JSON config will be silently untracked.
- ~91 MB of pickles are committed as raw git blobs (no LFS).
- No `LICENSE` file is present despite the MIT intent.

---

## 📚 Citation

```bibtex
@software{msigdb_signature_generator,
  author  = {Viswanathan, Arun Geetha},
  title   = {MSigDB Signature Generator},
  year    = {2026},
  doi     = {10.5281/zenodo.19386462},
  url     = {https://github.com/arunviswanathan91/msigdb-signature-generator}
}
```

## 🙏 Acknowledgments

- **MSigDB** (Broad Institute) — pathway gene sets
- **Groq** — LLM inference
- **sentence-transformers** — `all-MiniLM-L6-v2` embeddings
- **Streamlit** — application framework
