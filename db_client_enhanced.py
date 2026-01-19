"""
Database Client Enhanced - FIXED FOR EXISTING ENDPOINTS
========================================================

Version: 4.1.0 - Client-Side Validation Construction

CHANGES:
- Removed dependency on non-existent /api/debate/ endpoints
- Builds validation context using existing endpoints
- Aggregates data client-side instead of server-side
- Maintains same interface for backward compatibility
"""

import requests
from typing import List, Dict, Set, Optional, Tuple
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from collections import defaultdict


class DatabaseClientEnhanced:
    """
    Enhanced client for debate-driven signature generation.
    
    FIXED: Works with original db_client.py endpoints only.
    Constructs debate validation context client-side.
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
    # DEBATE INJECTOR METHODS (REWRITTEN TO USE EXISTING ENDPOINTS)
    # ============================================================
    
    def probe_signature_for_validation(
        self,
        genes: List[str],
        tissue_context: Optional[str] = None,
        min_expression_threshold: float = 1.0
    ) -> Dict:
        """
        Probe entire signature for debate validation.
        
        FIXED: Constructs validation using existing endpoints:
        - /api/expression/{gene} for GTEx data
        - /api/gene-neighbors/{gene} for connectivity
        - /api/evidence/{gene} for literature
        
        Args:
            genes: List of gene symbols
            tissue_context: Optional tissue for expression filtering
            min_expression_threshold: Minimum TPM
        
        Returns:
            {
                "signature_size": int,
                "issues": {
                    "housekeeping_genes": [],
                    "low_expression_genes": [],
                    "poorly_connected_genes": [],
                    "weak_evidence_genes": []
                },
                "sources_used": ["gtex_expression_v10.pkl.gz", ...]
            }
        """
        print(f"🔍 Analyzing {len(genes)} genes using existing endpoints...")
        
        issues = {
            "housekeeping_genes": [],
            "low_expression_genes": [],
            "poorly_connected_genes": [],
            "weak_evidence_genes": []
        }
        
        sources = set()
        
        # Known housekeeping genes (hard-coded since server doesn't have this)
        HOUSEKEEPING = {
            'GAPDH', 'ACTB', 'B2M', 'PPIA', 'PPIB', 'RPLP0', 
            'RPL13A', 'RPS18', 'HPRT1', 'TBP', 'YWHAZ', 'UBC'
        }
        
        for gene in genes:
            # Check 1: Housekeeping
            if gene in HOUSEKEEPING or gene.startswith(('RPL', 'RPS', 'MT-')):
                issues["housekeeping_genes"].append(gene)
            
            # Check 2: Expression (if tissue specified)
            if tissue_context:
                try:
                    expr_data = self.get_gene_expression(gene)
                    if expr_data:
                        sources.add("gtex_expression_v10.pkl.gz")
                        
                        # Check tissue-specific expression
                        tissues = expr_data.get('tissues', {})
                        
                        # Normalize tissue name for lookup
                        tissue_key = self._normalize_tissue_name(tissue_context, list(tissues.keys()))
                        
                        if tissue_key:
                            tpm = tissues.get(tissue_key, 0.0)
                            if tpm < min_expression_threshold:
                                issues["low_expression_genes"].append(gene)
                except:
                    pass  # Gene not in GTEx - skip
            
            # Check 3: Connectivity
            try:
                neighbors = self.get_gene_neighbors(gene, min_prob=0.05, top_k=10)
                if neighbors:
                    sources.add("gene_gene.db")
                    
                    if len(neighbors) < 3:  # Poorly connected
                        issues["poorly_connected_genes"].append(gene)
            except:
                # If we can't get neighbors, assume poorly connected
                issues["poorly_connected_genes"].append(gene)
            
            # Check 4: Evidence
            try:
                evidence = self.get_gene_evidence(gene)
                if evidence:
                    sources.add("gene_evidence.db")
                    
                    strength = evidence.get('evidence_strength', 'weak')
                    if strength == 'weak' or evidence.get('pathway_count', 0) < 5:
                        issues["weak_evidence_genes"].append(gene)
                else:
                    issues["weak_evidence_genes"].append(gene)
            except:
                issues["weak_evidence_genes"].append(gene)
        
        print(f"✅ Analysis complete. Found {sum(len(v) for v in issues.values())} total issues.")
        
        return {
            "signature_size": len(genes),
            "issues": issues,
            "sources_used": list(sources)
        }
    
    def get_expansion_candidate_context(
        self,
        candidate_gene: str,
        existing_genes: List[str]
    ) -> Dict:
        """
        Get detailed context for expansion debate.
        
        FIXED: Uses existing endpoints to build context.
        
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
                "evidence": {...},
                "sources": [...]
            }
        """
        print(f"🔍 Analyzing candidate gene: {candidate_gene}")
        
        sources = []
        
        # Get neighbors
        try:
            neighbors = self.get_gene_neighbors(candidate_gene, min_prob=0.01, top_k=50)
            sources.append("gene_gene.db")
            
            neighbor_genes = [n['target'] for n in neighbors]
            overlap_genes = [g for g in neighbor_genes if g in existing_genes]
            
        except:
            neighbors = []
            neighbor_genes = []
            overlap_genes = []
        
        # Get pathway associations
        try:
            pathways = self.get_gene_pathway_associations(candidate_gene, min_prob=0.02, top_k=20)
            sources.append("gene_pathway.db")
        except:
            pathways = []
        
        # Get expression
        try:
            expr_data = self.get_gene_expression(candidate_gene)
            sources.append("gtex_expression_v10.pkl.gz")
            
            if expr_data and 'tissues' in expr_data:
                # Get top 5 expressed tissues
                tissue_tpms = sorted(
                    expr_data['tissues'].items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )[:5]
                
                top_tissues = [
                    {"tissue": tissue, "tpm": tpm} 
                    for tissue, tpm in tissue_tpms
                ]
            else:
                top_tissues = []
        except:
            top_tissues = []
        
        # Get evidence
        try:
            evidence = self.get_gene_evidence(candidate_gene)
            sources.append("gene_evidence.db")
        except:
            evidence = {
                "pathway_count": 0,
                "literature_count": 0,
                "evidence_strength": "unknown",
                "pathway_memberships": []
            }
        
        result = {
            "candidate_gene": candidate_gene,
            "neighbors_total": len(neighbors),
            "neighbors_in_signature": len(overlap_genes),
            "neighbor_overlap_genes": overlap_genes[:10],  # Top 10
            "top_pathways": pathways[:10],  # Top 10
            "top_tissues": top_tissues,
            "evidence": evidence,
            "sources": list(set(sources))
        }
        
        print(f"✅ Context built for {candidate_gene}")
        return result
    
    def batch_expansion_candidates(
        self,
        candidate_genes: List[str],
        existing_genes: List[str]
    ) -> Dict[str, Dict]:
        """Batch version of expansion candidate probing."""
        results = {}
        for gene in candidate_genes:
            try:
                results[gene] = self.get_expansion_candidate_context(gene, existing_genes)
            except Exception as e:
                print(f"⚠️ Failed to probe {gene}: {e}")
                results[gene] = {"error": str(e)}
        
        return results
    
    def _normalize_tissue_name(self, query: str, available_tissues: List[str]) -> Optional[str]:
        """
        Normalize tissue name for GTEx lookup.
        
        Args:
            query: User's tissue input (e.g., "Adipose Tissue", "Blood")
            available_tissues: Available tissue names from GTEx
        
        Returns:
            Matched tissue name or None
        """
        query_lower = query.lower()
        
        # Direct match
        for tissue in available_tissues:
            if tissue.lower() == query_lower:
                return tissue
        
        # Partial match
        for tissue in available_tissues:
            if query_lower in tissue.lower() or tissue.lower() in query_lower:
                return tissue
        
        # Common mappings
        mappings = {
            'blood': 'Whole_Blood',
            'pbmc': 'Whole_Blood',
            'adipose': 'Adipose_Subcutaneous',
            'fat': 'Adipose_Subcutaneous',
            'liver': 'Liver',
            'brain': 'Brain_Cortex',
            'lung': 'Lung',
            'intestine': 'Small_Intestine_Terminal_Ileum',
            'colon': 'Colon_Transverse',
            'pancreas': 'Pancreas',
            'kidney': 'Kidney_Cortex',
            'heart': 'Heart_Left_Ventricle',
            'muscle': 'Muscle_Skeletal',
            'skin': 'Skin_Sun_Exposed_Lower_leg'
        }
        
        for key, value in mappings.items():
            if key in query_lower:
                # Check if this tissue exists
                for tissue in available_tissues:
                    if value.lower() in tissue.lower():
                        return tissue
        
        return None
    
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
        """Smart signature expansion using probabilistic networks."""
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
        """Get GTEx expression data for a gene."""
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
        """Get GTEx expression for gene in specific tissue."""
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
        """Batch GTEx query."""
        response = self.session.post(
            f"{self.api_url}/api/expression/batch",
            json=genes,
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    
    def check_enrichment(self, genes: List[str]) -> Dict:
        """Check GSEA enrichment cache."""
        response = self.session.post(
            f"{self.api_url}/api/enrichment",
            json={'genes': genes},
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    
    def get_gene_evidence(self, gene: str) -> Optional[Dict]:
        """Get evidence data for gene."""
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
    
    def get_gene_neighbors(
        self,
        gene: str,
        min_prob: float = 0.01,
        top_k: int = 20
    ) -> List[Dict]:
        """Get gene neighbors from gene-gene network."""
        response = self.session.get(
            f"{self.api_url}/api/gene-neighbors/{gene}",
            params={"min_prob": min_prob, "top_k": top_k},
            timeout=10
        )
        response.raise_for_status()
        return response.json()
    
    def get_gene_pathway_associations(
        self,
        gene: str,
        min_prob: float = 0.02,
        top_k: int = 20
    ) -> List[Dict]:
        """Get gene-pathway associations."""
        response = self.session.get(
            f"{self.api_url}/api/gene-pathway-prob/{gene}",
            params={"min_prob": min_prob, "top_k": top_k},
            timeout=10
        )
        response.raise_for_status()
        return response.json()
    
    def get_stats(self) -> Dict:
        """Get database statistics"""
        response = self.session.get(
            f"{self.api_url}/api/stats",
            timeout=10
        )
        response.raise_for_status()
        return response.json()


# ============================================================
# CONVENIENCE FUNCTIONS (Updated)
# ============================================================

def format_validation_issues_for_debate(issues: Dict) -> str:
    """Format validation issues into human-readable text for LLM debate."""
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
    """Format expansion candidate context for LLM debate."""
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
        lines.append(f"   Strength: {evid.get('evidence_strength', 'unknown')}")
        lines.append(f"   Pathways: {evid.get('pathway_count', 0)}")
        lines.append(f"   Publications: {evid.get('literature_count', 0)}")
        if evid.get('pathway_memberships'):
            lines.append(f"   Sample pathways: {', '.join(evid['pathway_memberships'][:3])}")
        lines.append("")
    
    lines.append(f"🗄️ Data sources: {', '.join(context['sources'])}")
    
    return "\n".join(lines)
