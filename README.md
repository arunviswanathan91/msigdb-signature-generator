# 🧬 MSigDB Signature Generator

> Production-grade biological signature generation pipeline with modern UI and dual knowledge base management

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 Overview

MSigDB Signature Generator is a comprehensive bioinformatics pipeline for generating biological gene signatures from pathway databases with **dual knowledge base management** - use our curated KB or build your own from custom GMT files.

### ✨ Key Features

- 🧠 **LLM-Powered Query Decomposition** - Intelligent query parsing with Qwen 72B
- 📚 **Dual KB Management** - Use built-in KB or build custom from GMT files
- 🎯 **Quota-Based Signature Selection** - Fair allocation across biological facets
- 🧬 **Multi-Source Integration** - KEGG, Reactome, GO, WikiPathways, Hallmark, and more
- 🎨 **Modern Surface UI** - Beautiful Tailwind-inspired interface
- 📊 **Live Progress Tracking** - Real-time pipeline execution monitoring
- 💾 **Multiple Export Formats** - JSON, TXT, GMT compatible outputs

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/msigdb-signature-generator.git
cd msigdb-signature-generator

# Install dependencies
pip install -r requirements.txt

# Run application
streamlit run app.py
```

### First-Time Setup

1. **Get Hugging Face Token** (Required)
   - Visit [Hugging Face Settings](https://huggingface.co/settings/tokens)
   - Create a token with "Read" access
   - Enter token in the app sidebar

2. **Choose Knowledge Base Mode**
   
   **Option A: Use Built-in KB** (Recommended for most users)
   - Includes 19,000+ curated pathways
   - Multiple sources (KEGG, Reactome, GO, etc.)
   - Ready to use immediately
   
   **Option B: Build Custom KB**
   - Upload your own GMT files
   - Customize min/max gene filters
   - Perfect for specialized research

## 📖 Usage

### Using Built-in Knowledge Base

1. Navigate to **"📚 Knowledge Base"** tab
2. Click **"🗄️ Use Built-in KB"**
3. Verify KB is loaded (shows pathway count, genes, sources)
4. Proceed to Pipeline tab

### Building Custom Knowledge Base

1. Navigate to **"📚 Knowledge Base"** tab
2. Click **"⚙️ Build Custom KB"**
3. Configure filters:
   - Min genes per pathway (default: 5)
   - Max genes per pathway (default: 500)
4. Upload GMT files (drag & drop or browse)
5. Click **"🔨 Build Knowledge Base"**
6. Wait for processing (~30-60 seconds)
7. Verify build statistics
8. Proceed to Pipeline tab

### Running the Pipeline

1. Navigate to **"🧬 Pipeline"** tab
2. **Stage 1: Query & Target**
   - Enter biological query (e.g., "pancreatic cancer pathways")
   - Set target signature count (10-200)
3. **Stage 2: Design Controls** (Optional)
   - Adjust gene size constraints
   - Configure diversity settings
4. Click **"🚀 Generate Signatures"**
5. Monitor real-time progress
6. View results in **"📊 Results"** tab

### Downloading Results

Navigate to **"📊 Results"** tab:
- View signature table with filtering
- Explore facet/source distributions
- Download in JSON, TXT, or GMT format

## 🗂️ Project Structure

```
msigdb-signature-generator/
├── app.py                          # Main Streamlit application
├── kb_builder.py                   # Custom KB builder module
├── complete_module_replacements.py # Pipeline modules
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── data/                          # Data directory
│   └── knowledge_base.json.gz     # Built-in KB (9MB)
└── results/                       # Generated results (auto-created)
```

## 📦 Requirements

### System Requirements
- Python 3.8 or higher
- 4GB RAM minimum (8GB recommended)
- ~3GB disk space for data files

### Python Packages
See `requirements.txt` for complete list:
- streamlit >= 1.28.0
- sentence-transformers >= 2.2.2
- faiss-cpu >= 1.7.4
- huggingface-hub >= 0.19.0
- pandas >= 2.0.0

## 🎨 User Interface

### Modern Design Features
- **Tailwind-inspired styling** - Clean, modern aesthetic
- **Surface UI components** - Glassmorphism effects
- **Responsive layout** - Works on all screen sizes
- **Dark theme** - Easy on the eyes for long sessions
- **Smooth animations** - Polished interactions
- **Status indicators** - Clear visual feedback

### Tab Organization
1. **📚 Knowledge Base** - KB selection and management
2. **🧬 Pipeline** - Query input and execution
3. **📊 Results** - Results viewing and downloads

## 🔧 Configuration

### Pipeline Parameters

Adjust in **Stage 2: Design Controls**:

```python
# Gene size constraints
min_genes = 5          # Minimum genes per signature
max_genes = 300        # Maximum genes per signature

# Overlap thresholds
within_facet_overlap = 0.50   # Same facet tolerance
cross_facet_overlap = 0.25    # Different facet tolerance

# Diversity weights (auto-normalized)
within_facet_weight = 0.7
cross_facet_weight = 0.3
```

### Custom KB Filters

```python
min_genes = 5    # Pathways with <5 genes excluded
max_genes = 500  # Pathways with >500 genes excluded
```

## 📊 Example Queries

### Cancer Research
```
Pathways involved in pancreatic cancer progression and metastasis
```

### Immune System
```
T cell activation and differentiation pathways in autoimmune diseases
```

### Metabolism
```
Metabolic pathways dysregulated in diabetes and obesity
```

### Drug Response
```
Pathways associated with chemotherapy resistance in breast cancer
```

## 🐛 Troubleshooting

### "Knowledge base not found"
- Ensure `data/knowledge_base.json.gz` exists
- Check file permissions
- Try switching to Custom KB mode

### "Token validation failed"
- Verify token is copied correctly (no spaces)
- Ensure token has "Read" access
- Try generating a new token

### "Pipeline failed"
- Check internet connection (for LLM API)
- Verify KB is loaded successfully
- Review query for special characters

### Custom KB Build Fails
- Ensure GMT files are properly formatted
- Check files are tab-separated
- Verify file encoding is UTF-8

## 🤝 Contributing

We welcome contributions! Areas for improvement:
- Additional pathway sources
- New signature derivation methods
- UI/UX enhancements
- Performance optimizations

## 📄 License

MIT License - See [LICENSE](LICENSE) file

## 🙏 Acknowledgments

- **MSigDB** - Curated gene sets
- **Hugging Face** - LLM inference API
- **Streamlit** - Application framework
- **FAISS** - Vector similarity search
- **Sentence Transformers** - Semantic embeddings

## 📚 Citation

If you use this tool in your research, please cite:

```bibtex
@software{msigdb_signature_generator,
  title={MSigDB Signature Generator},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/msigdb-signature-generator}
}
```

## 📧 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/msigdb-signature-generator/issues)
- **Documentation**: [Wiki](https://github.com/yourusername/msigdb-signature-generator/wiki)
- **Email**: your.email@example.com

---

**Built with ❤️ for the biological research community** 🧬

*Making pathway analysis accessible, powerful, and beautiful*
