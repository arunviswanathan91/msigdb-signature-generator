"""
Database Client Enhanced - Debate Injector Support
====================================================

Version: 4.0.0

NEW FEATURES for debate system:
- probe_signature_for_validation(): Returns aggregate issues for debate
- get_expansion_candidate_context(): Detailed context for gene expansion
- Database source tracking for attribution
- Batch operations for efficiency

Based on uploaded db_client.py with enhancements.
"""

import requests
from typing import List, Dict, Set, Optional, Tuple
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class DatabaseClientEnhanced:
    """
    Enhanced client for debate-driven signature generation.
    
    Key additions:
    - Validation probing: Check entire signature quality
    - Expansion probing: Investigate candidate genes
    - Source tracking: Attribute data to specific databases
    """
    
    def __init__(self, api_url: str):
        """
        Args:
            api_url: URL of database server
                    e.g., "https://arunviswanathan91-msigdb-api.hf.space"
        """
        self.api_url = api_url.rstrip('/')
        
        # Setup session with retries
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
        """Test server availability"""
        try:
            response = self.session.get(f"{self.api_url}/", timeout=10)
            response.raise_for_status()
            data = response.json()
            print(f"✅ Connected to {self.api_url} (v{data.get('version', 'unknown')})")
        except Exception as e:
            print(f"❌ Database server offline: {e}")
            raise ConnectionError(f"Cannot connect to {self.api_url}")
    
    # ============================================================
    # DEBATE INJECTOR METHODS (NEW)
    # ============================================================
    
    def probe_signature_for_validation(
        self,
        genes: List[str],
        tissue_context: Optional[str] = None,
        min_expression_threshold: float = 1.0
    ) -> Dict:
        """
        Probe entire signature for debate validation.
        
        Returns aggregate issues across all genes:
        - Housekeeping genes
        - Low expression genes (in tissue)
        - Poorly connected genes
        - Weak evidence genes
        
        Args:
            genes: List of gene symbols
            tissue_context: Optional tissue for expression filtering
            min_expression_threshold: Minimum TPM
        
        Returns:
            {
                "signature_size": int,
                "issues": {
                    "housekeeping_genes": ["GAPDH", ...],
                    "low_expression_genes": ["GENE1", ...],
                    "poorly_connected_genes": ["GENE2", ...],
                    "weak_evidence_genes": ["GENE3", ...]
                },
                "sources_used": ["gtex_expression_v10.pkl.gz", ...]
            }
        """
        response = self.session.post(
            f"{self.api_url}/api/debate/validate-signature",
            json={
                "genes": genes,
                "tissue_context": tissue_context,
                "min_expression_threshold": min_expression_threshold
            },
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    
    def get_expansion_candidate_context(
        self,
        candidate_gene: str,
        existing_genes: List[str]
    ) -> Dict:
        """
        Get detailed context for expansion debate.
        
        Used when debating whether to ADD a candidate gene.
        
        Args:
            candidate_gene: Gene to evaluate
            existing_genes: Current signature genes
        
        Returns:
            {
                "candidate_gene": "IL6R",
                "neighbors_total": 20,
                "neighbors_in_signature": 8,
                "neighbor_overlap_genes": ["IL6", "STAT3", ...],
                "top_pathways": [{"target": "JAK-STAT", "probability": 0.45}, ...],
                "top_tissues": [{"tissue": "Adipose", "tpm": 25.4}, ...],
                "evidence": {
                    "pathway_count": 12,
                    "literature_count": 5,
                    "evidence_strength": "moderate",
                    "pathway_memberships": ["pathway1", ...]
                },
                "sources": ["gene_gene.db", "gtex_expression_v10.pkl.gz", ...]
            }
        """
        response = self.session.post(
            f"{self.api_url}/api/debate/expansion-candidate",
            json={
                "candidate_gene": candidate_gene,
                "existing_genes": existing_genes
            },
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    
    def batch_expansion_candidates(
        self,
        candidate_genes: List[str],
        existing_genes: List[str]
    ) -> Dict[str, Dict]:
        """
        Batch version of expansion candidate probing.
        
        More efficient for debating multiple genes.
        
        Returns:
            {
                "IL6R": {...},
                "STAT3": {...},
                ...
            }
        """
        results = {}
        for gene in candidate_genes:
            try:
                results[gene] = self.get_expansion_candidate_context(gene, existing_genes)
            except Exception as e:
                print(f"⚠️ Failed to probe {gene}: {e}")
                results[gene] = {"error": str(e)}
        
        return results
    
    # ============================================================
    # ORIGINAL METHODS (from db_client.py)
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
        
        (Original method from db_client.py)
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
    
    def get_gene_expression(self, gene: str) -> Optional[Dict]:
        """
        Get GTEx expression data for a gene.
        
        Returns:
            {
                "gene": "IL17A",
                "tissues": {"Colon_Transverse_Mucosa": 0.016911, ...},
                "tissue_count": 56
            }
        """
        try:
            response = self.session.get(
                f"{self.api_url}/api/expression/{gene}",
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
        Get GTEx expression for gene in specific tissue.
        
        Returns:
            {"gene": "IL17A", "tissue": "...", "tpm": 0.016911}
        """
        try:
            response = self.session.get(
                f"{self.api_url}/api/expression/{gene}/{tissue}",
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
        Batch GTEx query.
        
        Returns:
            {
                "results": {"IL17A": {...}, "TNF": {...}},
                "found": 2,
                "not_found": ["FAKEGENE"]
            }
        """
        response = self.session.post(
            f"{self.api_url}/api/expression/batch",
            json=genes,
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    
    def check_enrichment(self, genes: List[str]) -> Dict:
        """
        Check GSEA enrichment cache.
        
        Returns:
            {
                "cached": true/false,
                "gene_count": 10,
                "pathways": [{"term": "...", "pval": 0.01}, ...]
            }
        """
        response = self.session.post(
            f"{self.api_url}/api/enrichment",
            json={'genes': genes},
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    
    def get_gene_evidence(self, gene: str) -> Optional[Dict]:
        """
        Get evidence data for gene.
        
        Returns:
            {
                "pathway_memberships": [...],
                "pathway_count": 221,
                "literature_pmids": [...],
                "literature_count": 5,
                "evidence_strength": "strong"
            }
        """
        try:
            response = self.session.get(
                f"{self.api_url}/api/evidence/{gene}",
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                return None
            raise
    
    def get_stats(self) -> Dict:
        """Get database statistics"""
        response = self.session.get(
            f"{self.api_url}/api/stats",
            timeout=10
        )
        response.raise_for_status()
        return response.json()


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================

def format_validation_issues_for_debate(issues: Dict) -> str:
    """
    Format validation issues into human-readable text for LLM debate.
    
    Args:
        issues: Result from probe_signature_for_validation()
    
    Returns:
        Formatted string for debate context
    """
    lines = ["📊 DATABASE VALIDATION RESULTS:", ""]
    
    if issues["issues"]["housekeeping_genes"]:
        genes_list = ", ".join(issues["issues"]["housekeeping_genes"])
        lines.append(f"⚠️ HOUSEKEEPING GENES ({len(issues['issues']['housekeeping_genes'])}):")
        lines.append(f"   {genes_list}")
        lines.append("   → These genes show constitutive expression across tissues")
        lines.append("")
    
    if issues["issues"]["low_expression_genes"]:
        genes_list = ", ".join(issues["issues"]["low_expression_genes"])
        lines.append(f"⚠️ LOW EXPRESSION GENES ({len(issues['issues']['low_expression_genes'])}):")
        lines.append(f"   {genes_list}")
        lines.append("   → Minimal expression in target tissue context")
        lines.append("")
    
    if issues["issues"]["poorly_connected_genes"]:
        genes_list = ", ".join(issues["issues"]["poorly_connected_genes"])
        lines.append(f"⚠️ POORLY CONNECTED GENES ({len(issues['issues']['poorly_connected_genes'])}):")
        lines.append(f"   {genes_list}")
        lines.append("   → Few functional associations with other genes")
        lines.append("")
    
    if issues["issues"]["weak_evidence_genes"]:
        genes_list = ", ".join(issues["issues"]["weak_evidence_genes"])
        lines.append(f"⚠️ WEAK EVIDENCE GENES ({len(issues['issues']['weak_evidence_genes'])}):")
        lines.append(f"   {genes_list}")
        lines.append("   → Limited pathway annotations or literature support")
        lines.append("")
    
    lines.append(f"🗄️ Data sources: {', '.join(issues['sources_used'])}")
    
    return "\n".join(lines)


def format_expansion_context_for_debate(context: Dict) -> str:
    """
    Format expansion candidate context for LLM debate.
    
    Args:
        context: Result from get_expansion_candidate_context()
    
    Returns:
        Formatted string for debate
    """
    gene = context["candidate_gene"]
    lines = [f"📊 CANDIDATE GENE ANALYSIS: {gene}", ""]
    
    # Network connectivity
    lines.append(f"🔗 NETWORK CONNECTIVITY:")
    lines.append(f"   Total neighbors: {context['neighbors_total']}")
    lines.append(f"   Overlap with existing signature: {context['neighbors_in_signature']} genes")
    if context['neighbor_overlap_genes']:
        overlap_str = ", ".join(context['neighbor_overlap_genes'][:5])
        if len(context['neighbor_overlap_genes']) > 5:
            overlap_str += "..."
        lines.append(f"   Shared genes: {overlap_str}")
    lines.append("")
    
    # Pathways
    if context['top_pathways']:
        lines.append(f"🛤️ TOP PATHWAYS:")
        for pw in context['top_pathways'][:5]:
            lines.append(f"   - {pw['target']} (prob: {pw['probability']:.3f})")
        lines.append("")
    
    # Expression
    if context['top_tissues']:
        lines.append(f"📈 TOP EXPRESSION:")
        for tissue_data in context['top_tissues']:
            lines.append(f"   - {tissue_data['tissue']}: {tissue_data['tpm']:.2f} TPM")
        lines.append("")
    
    # Evidence
    if context.get('evidence'):
        evid = context['evidence']
        lines.append(f"📚 LITERATURE EVIDENCE:")
        lines.append(f"   Strength: {evid['evidence_strength']}")
        lines.append(f"   Pathways: {evid['pathway_count']}")
        lines.append(f"   Publications: {evid['literature_count']}")
        if evid['pathway_memberships']:
            lines.append(f"   Sample pathways: {', '.join(evid['pathway_memberships'][:3])}")
        lines.append("")
    
    lines.append(f"🗄️ Data sources: {', '.join(context['sources'])}")
    
    return "\n".join(lines)
