# 🧬 MSigDB Signature Generator

> Production-grade biological signature generation pipeline with glassmorphism Streamlit UI

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 Overview

MSigDB Signature Generator is a comprehensive bioinformatics pipeline for generating biological gene signatures from pathway databases.

**Key Features:**
- 🧠 LLM-Powered Query Decomposition
- 🎯 Quota-Based Signature Selection  
- 🧬 Multi-Source Integration (KEGG, Reactome, GO, WikiPathways)
- 🎨 Beautiful Glassmorphism UI
- 📊 Live Progress Tracking
- 💾 Multiple Export Formats (JSON, TXT, GMT)

## 🚀 Quick Start

```bash
# Clone and install
git clone https://github.com/yourusername/msigdb-signature-generator.git
cd msigdb-signature-generator
pip install -r requirements.txt

# Run application
streamlit run app.py
```

See [GITHUB_SETUP_GUIDE.md](GITHUB_SETUP_GUIDE.md) for complete setup instructions.

## 📖 Documentation

- [User Guide](docs/USER_GUIDE.md) - How to use the application
- [GitHub Setup](GITHUB_SETUP_GUIDE.md) - Complete directory structure
- [Code Review](docs/code_review_changes.md) - Technical details

## 🗂️ Project Structure

```
msigdb-signature-generator/
├── app.py                    # Streamlit application
├── requirements.txt          # Dependencies
├── src/                      # Source code modules
├── data/                     # Data files (local only)
├── results/                  # Generated results
└── docs/                     # Documentation
```

## 📦 Requirements

- Python 3.8+
- Hugging Face account (for API token)
- ~3GB storage for data files

## 🎮 Usage

1. Enter biological query
2. Configure parameters
3. Run pipeline
4. Download results

See full guide in [docs/USER_GUIDE.md](docs/USER_GUIDE.md)

## 📝 License

MIT License - See [LICENSE](LICENSE)

## 🙏 Acknowledgments

MSigDB, Hugging Face, Streamlit, FAISS

---

**Built for biological research** 🧬
