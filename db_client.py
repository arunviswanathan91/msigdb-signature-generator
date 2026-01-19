"""
MSigDB Database Client
======================

Python client for interacting with the MSigDB API server.
"""

import requests
from typing import List, Dict, Optional, Set


class DatabaseClient:
    """Client for MSigDB remote database API"""
    
    def __init__(self, api_url: str):
        """
        Initialize database client.
        
        Args:
            api_url: Base URL of the API server (e.g., https://your-space.hf.space)
        """
        self.api_url = api_url.rstrip('/')
        self.session = requests.Session()
    
    def _get(self, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """Make GET request to API."""
        url = f"{self.api_url}{endpoint}"
        response = self.session.get(url, params=params, timeout=30)
        response.raise_for_status()
        return response.json()
    
    def _post(self, endpoint: str, data: Dict) -> Dict:
        """Make POST request to API."""
        url = f"{self.api_url}{endpoint}"
        response = self.session.post(url, json=data, timeout=30)
        response.raise_for_status()
        return response.json()
    
    def health_check(self) -> Dict:
        """Check API health status."""
        return self._get("/")
    
    def get_stats(self) -> Dict:
        """Get database statistics."""
        return self._get("/api/stats")
    
    def get_pathways_for_gene(self, gene: str) -> List[str]:
        """Get pathways containing a gene."""
        return self._get(f"/api/pathways-for-gene/{gene}")
    
    def get_genes_for_pathway(self, pathway: str) -> List[str]:
        """Get genes in a pathway."""
        return self._get(f"/api/genes-for-pathway/{pathway}")
    
    def get_gene_neighbors(
        self,
        gene: str,
        min_prob: float = 0.01,
        top_k: int = 20
    ) -> List[Dict]:
        """
        Get gene neighbors from gene-gene network.
        
        Args:
            gene: Gene symbol
            min_prob: Minimum probability threshold
            top_k: Maximum number of neighbors to return
            
        Returns:
            List of dicts with keys: target, probability, rank
        """
        return self._get(
            f"/api/gene-neighbors/{gene}",
            params={"min_prob": min_prob, "top_k": top_k}
        )
    
    def get_gene_pathway_associations(
        self,
        gene: str,
        min_prob: float = 0.02,
        top_k: int = 20
    ) -> List[Dict]:
        """
        Get gene-pathway associations.
        
        Returns:
            List of dicts with keys: target (pathway), probability, rank
        """
        return self._get(
            f"/api/gene-pathway-prob/{gene}",
            params={"min_prob": min_prob, "top_k": top_k}
        )
    
    def get_similar_pathways(
        self,
        pathway: str,
        min_prob: float = 0.02,
        top_k: int = 10
    ) -> List[Dict]:
        """Get similar pathways."""
        return self._get(
            f"/api/similar-pathways/{pathway}",
            params={"min_prob": min_prob, "top_k": top_k}
        )
    
    def expand_signature_smart(
        self,
        seed_genes: List[str],
        strength: float = 0.5,
        max_pathways_per_gene: int = 10
    ) -> Set[str]:
        """
        Expand gene signature using gene-gene network.
        
        Args:
            seed_genes: Initial gene list
            strength: Expansion strength (0-1)
            max_pathways_per_gene: Max neighbors per gene
            
        Returns:
            Set of expanded genes (includes seed genes)
        """
        min_prob = max(0.01, strength * 0.1)  # Convert strength to probability threshold
        
        result = self._post("/api/expand-signature", {
            "seed_genes": seed_genes,
            "strength": strength,
            "min_prob": min_prob,
            "max_neighbors": max_pathways_per_gene
        })
        
        return set(result.get("expanded_genes", seed_genes))
    
    def get_gene_expression(self, gene: str) -> Optional[Dict]:
        """
        Get GTEx expression data for a gene.
        
        Returns:
            Dict with keys: gene, tissues (dict of tissue->TPM), tissue_count
        """
        try:
            return self._get(f"/api/expression/{gene}")
        except requests.HTTPError as e:
            if e.response.status_code == 404:
                return None
            raise
    
    def get_gene_expression_in_tissue(
        self,
        gene: str,
        tissue: str
    ) -> Optional[float]:
        """
        Get expression of gene in specific tissue.
        
        Returns:
            TPM value or None if not found
        """
        try:
            result = self._get(f"/api/expression/{gene}/{tissue}")
            return result.get("tpm")
        except requests.HTTPError as e:
            if e.response.status_code == 404:
                return None
            raise
    
    def get_expression_batch(self, genes: List[str]) -> Dict:
        """
        Get expression for multiple genes.
        
        Returns:
            Dict with keys: results (dict of gene->tissues), found, not_found
        """
        return self._post("/api/expression/batch", genes)
    
    def check_enrichment(self, genes: List[str]) -> Dict:
        """
        Check for cached enrichment results.
        
        Returns:
            Dict with keys: cached (bool), pathways (if cached), gene_count
        """
        return self._post("/api/enrichment", {"genes": genes})
    
    def get_gene_evidence(self, gene: str) -> Optional[Dict]:
        """
        Get literature evidence for gene.
        
        Returns:
            Dict with keys: pathway_count, literature_count, evidence_strength, etc.
        """
        try:
            return self._get(f"/api/evidence/{gene}")
        except requests.HTTPError as e:
            if e.response.status_code == 404:
                return None
            raise
    
    def get_evidence_batch(self, genes: List[str]) -> Dict:
        """
        Get evidence for multiple genes.
        
        Returns:
            Dict with keys: results (dict), found, not_found
        """
        return self._post("/api/evidence/batch", genes)
