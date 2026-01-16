"""
Database Client - CORRECTED Version (v3.0)
==========================================
Fixed endpoint names to match db_server.py

CORRECTED ENDPOINTS:
- /api/expression/{gene} (was: gene-expression)
- /api/expression/batch (was: gene-expression/batch)
- /api/enrichment (was: gsea-lookup)
- /api/evidence/{gene} (was: gene-evidence)
- /api/evidence/batch (was: gene-evidence/batch)
"""

import requests
import streamlit as st
from typing import List, Dict, Set, Optional
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class DatabaseClient:
    """
    Enhanced client for remote database API.
    Provides access to SQLite databases + pickle files (GTEx, GSEA, Evidence).
    """
    
    def __init__(self, api_url: str):
        """
        Args:
            api_url: URL of your database server
                    e.g., "https://arunviswanathan91-msigdb-api.hf.space"
        """
        self.api_url = api_url.rstrip('/')
        
        # Setup session with retry logic
        self.session = requests.Session()
        retry = Retry(
            total=3,
            backoff_factor=0.3,
            status_forcelist=[500, 502, 503, 504]
        )
        adapter = HTTPAdapter(max_retries=retry)
        self.session.mount('http://', adapter)
        self.session.mount('https://', adapter)
        
        # Test connection
        self._test_connection()
    
    def _test_connection(self):
        """Test if server is reachable"""
        try:
            response = self.session.get(f"{self.api_url}/", timeout=10)
            response.raise_for_status()
            data = response.json()
            
            # Show enhanced data availability
            enhanced = data.get('enhanced', {})
            status_msg = f"🟢 Database connected (v{data.get('version', 'unknown')})\n"
            status_msg += f"  • GTEx: {enhanced.get('gtex_genes', 0):,} genes\n"
            status_msg += f"  • GSEA: {enhanced.get('gsea_entries', 0):,} entries\n"
            status_msg += f"  • Evidence: {enhanced.get('evidence_genes', 0):,} genes"
            
            st.sidebar.success(status_msg)
        except Exception as e:
            st.sidebar.error(f"🔴 Database offline: {e}")
            st.error("Cannot connect to database server. Please check API URL.")
            st.stop()
    
    # ============================================================
    # EXISTING METHODS (from original db_client.py)
    # ============================================================
    
    def expand_signature_smart(
        self,
        seed_genes: List[str],
        strength: float = 0.5,
        max_pathways_per_gene: int = 5,
        min_pathway_prob: float = 0.05,
        min_gene_prob: float = 0.05
    ) -> Set[str]:
        """
        Smart signature expansion using probabilistic networks.
        
        Args:
            seed_genes: Initial gene list
            strength: Expansion strength (0.0-1.0)
            max_pathways_per_gene: Max pathways to consider per gene
            min_pathway_prob: Minimum pathway probability threshold
            min_gene_prob: Minimum gene probability threshold
        
        Returns:
            Set of expanded genes (includes seed genes)
        """
        params = {
            'strength': strength,
            'max_pathways_per_gene': max_pathways_per_gene,
            'min_pathway_prob': min_pathway_prob,
            'min_gene_prob': min_gene_prob
        }
        
        response = self.session.post(
            f"{self.api_url}/api/expand-signature",
            json={'seed_genes': seed_genes},
            params=params,
            timeout=30
        )
        response.raise_for_status()
        result = response.json()
        return set(result['expanded_genes'])
    
    def get_stats(self) -> Dict:
        """Get comprehensive database statistics"""
        response = self.session.get(
            f"{self.api_url}/api/stats",
            timeout=10
        )
        response.raise_for_status()
        return response.json()
    
    # ============================================================
    # CORRECTED METHODS - GTEx EXPRESSION
    # ============================================================
    
    def get_gene_expression(self, gene: str) -> Optional[Dict]:
        """
        Get GTEx expression data for a gene across all tissues.
        
        Args:
            gene: Gene symbol (e.g., "IL17A")
        
        Returns:
            {
                "gene": "IL17A",
                "tissues": {"Colon_Transverse_Mucosa": 0.016911, ...},
                "tissue_count": 56
            }
            or None if gene not found
        """
        try:
            response = self.session.get(
                f"{self.api_url}/api/expression/{gene}",  # CORRECTED
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                return None
            raise
    
    def get_gene_expression_in_tissue(self, gene: str, tissue: str) -> Optional[Dict]:
        """
        Get GTEx expression for a gene in a specific tissue.
        
        Args:
            gene: Gene symbol
            tissue: Tissue name (e.g., "Adipose Tissue", "Liver")
        
        Returns:
            {"gene": "IL17A", "tissue": "...", "tpm": 0.016911}
            or None if not found
        """
        try:
            response = self.session.get(
                f"{self.api_url}/api/expression/{gene}/{tissue}",  # CORRECTED
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                return None
            raise
    
    def get_gene_expression_batch(self, genes: List[str]) -> Dict:
        """
        Get GTEx expression for multiple genes (batch query).
        
        Args:
            genes: List of gene symbols
        
        Returns:
            {
                "results": {"IL17A": {...}, "TNF": {...}},
                "found": 2,
                "not_found": ["FAKEGENE"]
            }
        """
        response = self.session.post(
            f"{self.api_url}/api/expression/batch",  # CORRECTED
            json=genes,
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    
    # ============================================================
    # CORRECTED METHODS - GSEA CACHE
    # ============================================================
    
    def check_enrichment(self, genes: List[str]) -> Dict:
        """
        Check GSEA enrichment cache.
        
        Args:
            genes: List of gene symbols
        
        Returns:
            {
                "cached": true/false,
                "gene_count": 10,
                "pathways": [{"term": "...", "pval": 0.01, "fdr": 0.05}, ...]
            }
        """
        response = self.session.post(
            f"{self.api_url}/api/enrichment",  # CORRECTED (was gsea-lookup)
            json={'genes': genes},
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    
    # ============================================================
    # CORRECTED METHODS - GENE EVIDENCE
    # ============================================================
    
    def get_gene_evidence(self, gene: str) -> Optional[Dict]:
        """
        Get evidence data for a gene (pathway memberships, literature, strength).
        
        Args:
            gene: Gene symbol
        
        Returns:
            {
                "gene": "IL17A",
                "pathway_memberships": ["Interleukin-17 signaling", ...],
                "pathway_count": 221,
                "literature_pmids": ["41523289", ...],
                "literature_count": 5,
                "evidence_strength": "strong"
            }
            or None if not found
        """
        try:
            response = self.session.get(
                f"{self.api_url}/api/evidence/{gene}",  # CORRECTED
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                return None
            raise
    
    def get_gene_evidence_batch(self, genes: List[str]) -> Dict:
        """
        Get evidence for multiple genes (batch query).
        
        Args:
            genes: List of gene symbols
        
        Returns:
            {
                "results": {"IL17A": {...}, "TNF": {...}},
                "found": 2,
                "not_found": ["FAKEGENE"]
            }
        """
        response = self.session.post(
            f"{self.api_url}/api/evidence/batch",  # CORRECTED
            json=genes,
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    
    # ============================================================
    # CONVENIENCE METHODS
    # ============================================================
    
    def filter_by_tissue_expression(
        self,
        genes: List[str],
        tissue: str,
        min_tpm: float = 1.0
    ) -> tuple[List[str], List[str]]:
        """
        Filter genes by expression level in a specific tissue.
        
        Args:
            genes: List of gene symbols
            tissue: Tissue name
            min_tpm: Minimum TPM threshold
        
        Returns:
            (kept_genes, removed_genes)
        """
        batch_result = self.get_gene_expression_batch(genes)
        
        kept = []
        removed = []
        
        for gene in genes:
            if gene in batch_result['results']:
                tissues_data = batch_result['results'][gene]
                
                # Try exact match
                tpm = tissues_data.get(tissue, 0.0)
                
                # Try partial match if exact fails
                if tpm == 0.0:
                    for tissue_name, tpm_val in tissues_data.items():
                        if tissue.lower() in tissue_name.lower():
                            tpm = tpm_val
                            break
                
                if tpm >= min_tpm:
                    kept.append(gene)
                else:
                    removed.append(gene)
            else:
                removed.append(gene)
        
        return kept, removed
    
    def filter_by_evidence_strength(
        self,
        genes: List[str],
        min_strength: str = 'moderate'
    ) -> tuple[List[str], List[str]]:
        """
        Filter genes by literature evidence strength.
        
        Args:
            genes: List of gene symbols
            min_strength: Minimum strength ('minimal', 'weak', 'moderate', 'strong')
        
        Returns:
            (kept_genes, removed_genes)
        """
        batch_result = self.get_gene_evidence_batch(genes)
        
        strength_order = {'minimal': 0, 'weak': 1, 'moderate': 2, 'strong': 3}
        threshold = strength_order.get(min_strength, 1)
        
        kept = []
        removed = []
        
        for gene in genes:
            if gene in batch_result['results']:
                evidence = batch_result['results'][gene]
                gene_strength = evidence.get('evidence_strength', 'minimal')
                
                if strength_order.get(gene_strength, 0) >= threshold:
                    kept.append(gene)
                else:
                    removed.append(gene)
            else:
                removed.append(gene)
        
        return kept, removed
    
    def validate_with_gsea(
        self,
        genes: List[str],
        min_pvalue: float = 0.05
    ) -> tuple[bool, Dict]:
        """
        Validate gene set using GSEA enrichment.
        
        Args:
            genes: List of gene symbols
            min_pvalue: P-value threshold
        
        Returns:
            (is_valid, enrichment_data)
        """
        enrichment = self.check_enrichment(genes)
        
        if not enrichment.get('cached'):
            return True, enrichment  # Can't validate if not cached
        
        pathways = enrichment.get('pathways', [])
        significant = [p for p in pathways if p.get('pval', 1.0) < min_pvalue]
        
        is_valid = len(significant) > 0
        
        return is_valid, enrichment
    
    def enrich_genes_with_context(
        self,
        genes: List[str],
        tissue: Optional[str] = None,
        include_enrichment: bool = True
    ) -> Dict:
        """
        Gather ALL context for genes (for debate prompts).
        
        Args:
            genes: List of gene symbols
            tissue: Optional tissue for expression
            include_enrichment: Whether to include GSEA
        
        Returns:
            {
                'expression': {gene: {tissue: tpm}},
                'tissue_specific': {gene: tpm} if tissue provided,
                'evidence': {gene: {pathway_count, pmids, ...}},
                'enrichment': {cached, pathways} if requested
            }
        """
        context = {}
        
        # Get expression
        expr_batch = self.get_gene_expression_batch(genes)
        context['expression'] = expr_batch.get('results', {})
        
        # Get tissue-specific if provided
        if tissue:
            tissue_expr = {}
            for gene, tissues_data in context['expression'].items():
                # Try exact match
                tpm = tissues_data.get(tissue, 0.0)
                
                # Try partial match
                if tpm == 0.0:
                    for tissue_name, tpm_val in tissues_data.items():
                        if tissue.lower() in tissue_name.lower():
                            tpm = tpm_val
                            break
                
                tissue_expr[gene] = tpm
            
            context['tissue_specific'] = tissue_expr
        
        # Get evidence
        evid_batch = self.get_gene_evidence_batch(genes)
        context['evidence'] = evid_batch.get('results', {})
        
        # Get enrichment if requested
        if include_enrichment:
            context['enrichment'] = self.check_enrichment(genes)
        
        return context


# ============================================================
# HELPER: FORMAT EVIDENCE FOR LLM PROMPTS
# ============================================================

def format_evidence_for_debate(
    genes: List[str],
    context: Dict,
    tissue: Optional[str] = None
) -> str:
    """
    Format database evidence into human-readable text for LLM prompts.
    
    Args:
        genes: List of gene symbols
        context: Context dict from enrich_genes_with_context()
        tissue: Optional tissue name
    
    Returns:
        Formatted evidence string
    """
    lines = ["DATABASE EVIDENCE:", ""]
    
    # Tissue expression
    if tissue and 'tissue_specific' in context:
        lines.append(f"GTEx Expression in {tissue}:")
        tissue_expr = context['tissue_specific']
        
        for gene in genes:
            tpm = tissue_expr.get(gene, 0.0)
            
            # Classify expression level
            if tpm >= 10:
                level = "high"
            elif tpm >= 1:
                level = "moderate"
            elif tpm > 0:
                level = "low"
            else:
                level = "not detected"
            
            lines.append(f"  {gene}: {tpm:.2f} TPM ({level})")
        
        lines.append("")
    
    # Literature evidence
    if 'evidence' in context:
        lines.append("Literature Evidence:")
        evidence = context['evidence']
        
        for gene in genes:
            gene_evid = evidence.get(gene, {})
            pathway_count = gene_evid.get('pathway_count', 0)
            lit_count = gene_evid.get('literature_count', 0)
            strength = gene_evid.get('evidence_strength', 'unknown')
            
            lines.append(
                f"  {gene}: {pathway_count} pathways, {lit_count} PMIDs, "
                f"strength={strength}"
            )
        
        lines.append("")
    
    # GSEA enrichment
    if 'enrichment' in context:
        enrichment = context['enrichment']
        
        if enrichment.get('cached'):
            lines.append("GSEA Enrichment (cached):")
            pathways = enrichment.get('pathways', [])[:5]  # Top 5
            
            for pw in pathways:
                term = pw.get('term', 'Unknown')
                pval = pw.get('pval', 1.0)
                fdr = pw.get('fdr', 1.0)
                lines.append(f"  - {term} (p={pval:.4f}, fdr={fdr:.4f})")
            
            lines.append("")
        else:
            lines.append("GSEA Enrichment: Not cached")
            lines.append("")
    
    return "\n".join(lines)
