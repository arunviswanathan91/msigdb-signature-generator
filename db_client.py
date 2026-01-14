"""
Database Client - Enhanced Version (v3.0)
==========================================
Now includes methods for GTEx expression, GSEA cache, and Gene Evidence!

NEW METHODS:
- get_gene_expression()
- get_gene_expression_in_tissue()
- get_gene_expression_batch()
- gsea_lookup()
- get_gene_evidence()
- get_gene_evidence_batch()
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
                    e.g., "https://username-msigdb-api.hf.space"
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
            enhanced = data.get('enhanced_data', {})
            status_msg = f"🟢 Database connected (v{data.get('version', 'unknown')})\n"
            status_msg += f"  • GTEx: {enhanced.get('gtex_expression', 0):,} genes\n"
            status_msg += f"  • GSEA: {enhanced.get('gsea_cache', 0):,} entries\n"
            status_msg += f"  • Evidence: {enhanced.get('gene_evidence', 0):,} genes"
            
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
            json=seed_genes,
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
    # NEW METHODS - GTEx EXPRESSION
    # ============================================================
    
    def get_gene_expression(self, gene: str) -> Optional[Dict]:
        """
        Get GTEx expression data for a gene across all tissues.
        
        Args:
            gene: Gene symbol (e.g., "IL17A")
        
        Returns:
            {
                "gene": "IL17A",
                "expression": {"Colon_Transverse_Mucosa": 0.016911, ...},
                "tissue_count": 56
            }
            or None if gene not found
        """
        try:
            response = self.session.get(
                f"{self.api_url}/api/gene-expression/{gene}",
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
            tissue: Tissue name (use underscores, e.g., "Colon_Transverse_Mucosa")
        
        Returns:
            {"gene": "IL17A", "tissue": "...", "tpm": 0.016911}
            or None if not found
        """
        try:
            response = self.session.get(
                f"{self.api_url}/api/gene-expression/{gene}/tissue/{tissue}",
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
            f"{self.api_url}/api/gene-expression/batch",
            json=genes,
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    
    # ============================================================
    # NEW METHODS - GSEA CACHE
    # ============================================================
    
    def gsea_lookup(self, gene_set: List[str], top_k: int = 10) -> Dict:
        """
        Look up GSEA enrichment results from cache.
        
        Args:
            gene_set: List of gene symbols
            top_k: Number of top pathways to return
        
        Returns:
            {
                "cache_hit": true/false,
                "gene_count": 10,
                "pathways": [{"name": "...", "score": 0.95}, ...]
            }
        """
        response = self.session.post(
            f"{self.api_url}/api/gsea-lookup",
            json=gene_set,
            params={'top_k': top_k},
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    
    # ============================================================
    # NEW METHODS - GENE EVIDENCE
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
                f"{self.api_url}/api/gene-evidence/{gene}",
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
            f"{self.api_url}/api/gene-evidence/batch",
            json=genes,
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    
    # ============================================================
    # CONVENIENCE METHODS
    # ============================================================
    
    def get_genes_with_high_expression(
        self,
        genes: List[str],
        tissue: str,
        min_tpm: float = 1.0
    ) -> List[str]:
        """
        Filter genes by expression level in a specific tissue.
        
        Args:
            genes: List of gene symbols to filter
            tissue: Tissue name
            min_tpm: Minimum TPM threshold
        
        Returns:
            List of genes with TPM >= min_tpm in the specified tissue
        """
        batch_result = self.get_gene_expression_batch(genes)
        
        high_expression_genes = []
        
        for gene, expression_data in batch_result['results'].items():
            if tissue in expression_data:
                tpm = expression_data[tissue]
                if tpm >= min_tpm:
                    high_expression_genes.append(gene)
        
        return high_expression_genes
    
    def get_genes_with_strong_evidence(
        self,
        genes: List[str]
    ) -> List[str]:
        """
        Filter genes by evidence strength.
        
        Args:
            genes: List of gene symbols to filter
        
        Returns:
            List of genes with "strong" evidence
        """
        batch_result = self.get_gene_evidence_batch(genes)
        
        strong_genes = []
        
        for gene, evidence_data in batch_result['results'].items():
            if evidence_data.get('evidence_strength') == 'strong':
                strong_genes.append(gene)
        
        return strong_genes
    
    def enrich_genes_with_context(
        self,
        genes: List[str],
        tissue: Optional[str] = None
    ) -> List[Dict]:
        """
        Enrich gene list with expression and evidence data.
        
        Args:
            genes: List of gene symbols
            tissue: Optional tissue to include expression for
        
        Returns:
            List of enriched gene dictionaries with all available data
        """
        # Get expression data
        expression_result = self.get_gene_expression_batch(genes)
        
        # Get evidence data
        evidence_result = self.get_gene_evidence_batch(genes)
        
        enriched_genes = []
        
        for gene in genes:
            gene_data = {'gene': gene}
            
            # Add expression
            if gene in expression_result['results']:
                if tissue:
                    expr = expression_result['results'][gene].get(tissue, 0.0)
                    gene_data['expression_tpm'] = expr
                else:
                    gene_data['expression'] = expression_result['results'][gene]
            
            # Add evidence
            if gene in evidence_result['results']:
                evidence = evidence_result['results'][gene]
                gene_data['evidence_strength'] = evidence.get('evidence_strength', 'unknown')
                gene_data['pathway_count'] = evidence.get('pathway_count', 0)
                gene_data['literature_count'] = evidence.get('literature_count', 0)
            
            enriched_genes.append(gene_data)
        
        return enriched_genes
