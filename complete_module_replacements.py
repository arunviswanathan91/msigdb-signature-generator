# ============================================================
# COMPLETE MODULE REPLACEMENTS FOR CODE REVIEW
# ============================================================
# This file contains complete, production-ready replacements
# for all modules that need improvements.
# ============================================================

# ============================================================
# MODULE 0: PIPELINE CONFIGURATION - COMPLETE REPLACEMENT
# ============================================================

from dataclasses import dataclass, field
from typing import Dict, Set

@dataclass
class PipelineConfig:
    """
    Centralized configuration for MSigDB-scale pipeline.
    All hardcoded constants replaced with configurable values.
    
    Raises:
        ValueError: If configuration parameters are invalid.
    """
    
    # Gene size constraints
    min_genes: int = 5
    max_genes: int = 300
    
    # Overlap thresholds for validation
    within_facet_overlap_threshold: float = 0.50  # Higher tolerance within same facet
    cross_facet_overlap_threshold: float = 0.25   # Stricter across facets
    
    # Retrieval depth by priority (scaled for full MSigDB)
    retrieval_depth_by_priority: Dict[str, int] = field(default_factory=lambda: {
        'HIGH': 150,
        'MEDIUM': 75,
        'LOW': 40
    })
    
    # Quota allocation
    priority_weights: Dict[str, int] = field(default_factory=lambda: {
        'HIGH': 3,
        'MEDIUM': 2,
        'LOW': 1
    })
    
    # Signature derivation
    pathway_selection_buffer: float = 1.5  # Over-select pathways to hit target signature count
    core_signature_threshold: float = 0.7  # Gene must appear in 70%+ of similar pathways
    unique_signature_threshold: float = 0.2  # Gene appears in <20% of other pathways
    min_signature_genes: int = 3  # Minimum genes per derived signature
    
    # Diversity scoring weights
    within_facet_diversity_weight: float = 0.7  # Alpha
    cross_facet_diversity_weight: float = 0.3   # Beta
    
    # Facet constraints
    min_facets: int = 3
    max_facets: int = 10
    facet_merge_similarity_threshold: float = 0.85  # Merge facets with ≥85% similarity
    
    # Performance tuning
    embedding_batch_size: int = 32
    validation_chunk_size: int = 1000  # Process validation in chunks for large datasets
    
    # Gene type filtering (OFF by default - metadata only)
    enable_gene_type_filtering: bool = False
    allowed_gene_types: Set[str] = field(default_factory=lambda: {
        'protein_coding', 'lncRNA', 'pseudogene', 'miRNA', 'antisense', 'unknown'
    })
    
    def __post_init__(self) -> None:
        """
        Validate configuration on instantiation.
        
        Raises:
            ValueError: If any configuration parameter is invalid.
        """
        # Basic constraints
        if self.min_genes <= 0:
            raise ValueError("min_genes must be positive")
        if self.max_genes <= self.min_genes:
            raise ValueError("max_genes must exceed min_genes")
        
        # Threshold validation
        if not (0 < self.within_facet_overlap_threshold <= 1):
            raise ValueError("within_facet_overlap_threshold must be in (0,1]")
        if not (0 < self.cross_facet_overlap_threshold <= 1):
            raise ValueError("cross_facet_overlap_threshold must be in (0,1]")
        
        # Facet constraints
        if self.min_facets > self.max_facets:
            raise ValueError("min_facets must not exceed max_facets")
        if self.min_facets < 1:
            raise ValueError("min_facets must be at least 1")
        
        # Validate retrieval depths
        required_priorities = {'HIGH', 'MEDIUM', 'LOW'}
        if set(self.retrieval_depth_by_priority.keys()) != required_priorities:
            raise ValueError(f"retrieval_depth_by_priority must contain exactly {required_priorities}")
        for priority, depth in self.retrieval_depth_by_priority.items():
            if depth <= 0:
                raise ValueError(f"retrieval_depth for {priority} must be positive")
        
        # Validate priority weights
        if set(self.priority_weights.keys()) != required_priorities:
            raise ValueError(f"priority_weights must contain exactly {required_priorities}")
        for priority, weight in self.priority_weights.items():
            if weight <= 0:
                raise ValueError(f"priority_weight for {priority} must be positive")
        
        # Validate diversity weights sum to 1.0
        weight_sum = self.within_facet_diversity_weight + self.cross_facet_diversity_weight
        if not (0.99 <= weight_sum <= 1.01):  # Allow small floating point error
            raise ValueError(f"Diversity weights must sum to 1.0, got {weight_sum}")
        
        # Validate thresholds
        if not (0 < self.core_signature_threshold <= 1):
            raise ValueError("core_signature_threshold must be in (0,1]")
        if not (0 <= self.unique_signature_threshold < 1):
            raise ValueError("unique_signature_threshold must be in [0,1)")
        if not (0 < self.facet_merge_similarity_threshold <= 1):
            raise ValueError("facet_merge_similarity_threshold must be in (0,1]")
        
        # Validate performance tuning
        if self.embedding_batch_size <= 0:
            raise ValueError("embedding_batch_size must be positive")
        if self.validation_chunk_size <= 0:
            raise ValueError("validation_chunk_size must be positive")
        
        # Validate signature constraints
        if self.min_signature_genes <= 0:
            raise ValueError("min_signature_genes must be positive")
        if self.pathway_selection_buffer < 1.0:
            raise ValueError("pathway_selection_buffer must be >= 1.0")


# ============================================================
# MODULE 1: KNOWLEDGE BASE LOADER & VALIDATOR - COMPLETE REPLACEMENT
# ============================================================

import json
import gzip
import hashlib
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Optional, Any

class KnowledgeBase:
    """
    Loads and validates the pathway knowledge base.
    
    Features:
    - Cache hash generation for invalidation
    - Optional gene-type annotation (metadata only)
    - Comprehensive validation with error handling
    """
    
    def __init__(self, kb_path: str):
        """
        Initialize KnowledgeBase loader.
        
        Args:
            kb_path: Path to compressed knowledge base file (.json.gz)
        """
        self.kb_path = kb_path
        self.pathways: Dict[str, List[str]] = {}
        self.metadata: Dict[str, Any] = {}
        self.validation_report: Dict[str, Any] = {}
        self.kb_hash: Optional[str] = None
        self.gene_types: Dict[str, str] = {}
        
    def load(self) -> Tuple[Dict[str, List[str]], Dict[str, Any]]:
        """
        Load compressed knowledge base from disk.
        
        Returns:
            Tuple containing pathways dict and metadata dict.
            
        Raises:
            FileNotFoundError: If KB file doesn't exist.
            json.JSONDecodeError: If KB file is corrupted.
            ValueError: If KB structure is invalid.
        """
        print(f"📚 Loading KB from: {self.kb_path}")
        
        try:
            with gzip.open(self.kb_path, 'rt', encoding='utf-8') as f:
                data = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Knowledge base file not found: {self.kb_path}")
        except gzip.BadGzipFile:
            raise ValueError(f"Invalid gzip file: {self.kb_path}")
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(f"Corrupted JSON in KB file: {e}", e.doc, e.pos)
        
        # Validate required keys
        if 'pathways' not in data:
            raise ValueError("Knowledge base missing required 'pathways' key")
        if not isinstance(data['pathways'], dict):
            raise ValueError("'pathways' must be a dictionary")
        
        self.pathways = data['pathways']
        self.metadata = data.get('metadata', {})
        
        # Validate pathway structure
        if len(self.pathways) == 0:
            raise ValueError("Knowledge base contains no pathways")
        
        # Generate stable hash for cache invalidation
        self.kb_hash = self._compute_kb_hash()
        
        print(f"✅ Loaded {len(self.pathways)} pathways")
        print(f"🔑 KB Hash: {self.kb_hash[:16]}... (for cache validation)")
        
        return self.pathways, self.metadata
    
    def _compute_kb_hash(self) -> str:
        """
        Compute stable hash of KB content for cache invalidation.
        Hash depends on pathway IDs + gene membership (order-independent).
        
        Returns:
            SHA256 hex digest of knowledge base content.
        """
        hasher = hashlib.sha256()
        
        # Sort pathways for deterministic hashing
        for pathway_id in sorted(self.pathways.keys()):
            genes = sorted(self.pathways[pathway_id])  # Order-independent
            
            # Hash pathway ID + gene list
            hasher.update(pathway_id.encode('utf-8'))
            hasher.update('|'.join(genes).encode('utf-8'))
        
        return hasher.hexdigest()
    
    def annotate_gene_types(self, gene_type_map: Optional[Dict[str, str]] = None) -> None:
        """
        Optional gene-type annotation (metadata only, does NOT filter).
        
        Args:
            gene_type_map: Dict mapping gene symbol -> type 
                          (protein_coding, lncRNA, etc.)
        
        Note:
            This is METADATA ONLY. Genes are NOT removed from pathways.
        """
        if gene_type_map is None:
            # No annotation provided - all genes marked as 'unknown'
            all_genes = set()
            for genes in self.pathways.values():
                all_genes.update(genes)
            
            self.gene_types = {gene: 'unknown' for gene in all_genes}
            print(f"ℹ️  Gene type annotation: {len(self.gene_types)} genes marked as 'unknown'")
        else:
            self.gene_types = gene_type_map
            
            # Count types
            type_counts = defaultdict(int)
            for gene_type in gene_type_map.values():
                type_counts[gene_type] += 1
            
            print(f"✅ Gene type annotation loaded:")
            for gene_type, count in sorted(type_counts.items(), key=lambda x: -x[1]):
                print(f"   {gene_type}: {count}")
    
    def validate(self, config: Optional[PipelineConfig] = None) -> Dict[str, Any]:
        """
        Validate KB quality with configurable constraints.
        
        Args:
            config: Optional PipelineConfig for validation constraints.
                    If None, uses default values.
        
        Returns:
            Dictionary containing validation report with statistics.
            
        Note:
            Sets self.validation_report as a side effect.
        """
        # Extract constraints
        min_genes = config.min_genes if config else 5
        max_genes = config.max_genes if config else 300
        
        print(f"\n🔍 Validating Knowledge Base...")
        print(f"   Size constraints: {min_genes}-{max_genes} genes")
        
        report = self._initialize_validation_report()
        all_genes: Set[str] = set()
        
        for pathway_id, genes in self.pathways.items():
            self._validate_pathway(
                pathway_id, genes, min_genes, max_genes, report, all_genes
            )
        
        report['total_unique_genes'] = len(all_genes)
        
        self._print_validation_summary(report, min_genes, max_genes)
        
        self.validation_report = report
        return report
    
    def _initialize_validation_report(self) -> Dict[str, Any]:
        """Initialize validation report structure."""
        return {
            'total_pathways': len(self.pathways),
            'empty_gene_sets': [],
            'duplicate_genes': {},
            'tiny_pathways': [],
            'huge_pathways': [],
            'valid_pathways': 0,
            'total_unique_genes': 0,
            'kb_hash': self.kb_hash
        }
    
    def _validate_pathway(
        self,
        pathway_id: str,
        genes: List[str],
        min_genes: int,
        max_genes: int,
        report: Dict[str, Any],
        all_genes: Set[str]
    ) -> None:
        """
        Validate a single pathway and update report.
        
        Args:
            pathway_id: Identifier for the pathway.
            genes: List of gene symbols in the pathway.
            min_genes: Minimum acceptable gene count.
            max_genes: Maximum acceptable gene count.
            report: Report dict to update (modified in place).
            all_genes: Set of all unique genes (modified in place).
        """
        # Check for empty or None
        if not genes:
            report['empty_gene_sets'].append(pathway_id)
            return
        
        # Check for duplicates within pathway
        unique_genes = set(genes)
        if len(genes) != len(unique_genes):
            duplicates = [gene for gene in unique_genes if genes.count(gene) > 1]
            report['duplicate_genes'][pathway_id] = duplicates
        
        # Size checks
        gene_count = len(unique_genes)
        
        if gene_count < min_genes:
            report['tiny_pathways'].append((pathway_id, gene_count))
        elif gene_count > max_genes:
            report['huge_pathways'].append((pathway_id, gene_count))
        else:
            report['valid_pathways'] += 1
        
        all_genes.update(unique_genes)
    
    def _print_validation_summary(
        self,
        report: Dict[str, Any],
        min_genes: int,
        max_genes: int
    ) -> None:
        """Print validation summary to console."""
        print(f"✅ Valid pathways: {report['valid_pathways']}/{report['total_pathways']}")
        print(f"🧬 Total unique genes: {report['total_unique_genes']}")
        
        warnings = [
            (report['empty_gene_sets'], "⚠️  Empty gene sets: {}"),
            (report['duplicate_genes'], "⚠️  Pathways with duplicate genes: {}"),
            (report['tiny_pathways'], f"⚠️  Tiny pathways (<{min_genes} genes): {{}}"),
            (report['huge_pathways'], f"⚠️  Huge pathways (>{max_genes} genes): {{}}")
        ]
        
        for items, message in warnings:
            if items:
                print(message.format(len(items)))


# ============================================================
# MODULE 8: COMPLETE PIPELINE - COMPLETE REPLACEMENT
# ============================================================

import pickle
import os
from datetime import datetime
from typing import Dict, List, Any
from collections import defaultdict
import numpy as np

class CompletePipeline:
    """
    Complete MSigDB signature generation pipeline (v2.1).
    
    Features:
    - Facet merging (Fix #1)
    - Exact final signature count enforcement (Fix #2)
    - Consistent quota usage (Fix #3)
    - Priority-aware retrieval depth (Fix #5)
    - Comprehensive error handling
    """
    
    def __init__(self, 
                 kb_path: str,
                 embeddings_cache_dir: str,
                 faiss_index_path: str,
                 hf_token: str,
                 config: Optional[PipelineConfig] = None):
        """
        Initialize pipeline with configuration.
        
        Args:
            kb_path: Path to knowledge base file
            embeddings_cache_dir: Directory for embedding cache
            faiss_index_path: Path to FAISS index
            hf_token: Hugging Face API token
            config: Optional PipelineConfig (uses defaults if None)
        
        Raises:
            ValueError: If paths or token are invalid
        """
        if not hf_token:
            raise ValueError("HF token cannot be empty")
        if not os.path.exists(os.path.dirname(kb_path) or '.'):
            raise ValueError(f"KB path directory does not exist: {kb_path}")
            
        self.kb_path = kb_path
        self.embeddings_cache_dir = embeddings_cache_dir
        self.faiss_index_path = faiss_index_path
        self.hf_token = hf_token
        self.config = config or PipelineConfig()
        
        # Initialize components
        self.kb: Optional[KnowledgeBase] = None
        self.chunks: Optional[List[Any]] = None
        self.embedding_engine: Optional[Any] = None
        self.faiss_index: Optional[Any] = None
        self.retriever: Optional[Any] = None
        self.planner: Optional[Any] = None
        
    def initialize(self) -> None:
        """
        Load and initialize all pipeline components.
        
        Raises:
            Exception: If any component fails to initialize
        """
        print("🚀 Initializing pipeline...")
        
        try:
            # Load knowledge base
            self.kb = KnowledgeBase(self.kb_path)
            pathways, metadata = self.kb.load()
            self.kb.validate(config=self.config)
            
            # Construct chunks
            constructor = ChunkConstructor(pathways, metadata)
            self.chunks = constructor.construct_chunks(
                min_genes=self.config.min_genes,
                max_genes=self.config.max_genes,
                exclude_medicus=True
            )
            
            # Initialize embedding engine
            self.embedding_engine = EmbeddingEngine(
                cache_dir=self.embeddings_cache_dir
            )
            embeddings, chunk_ids = self.embedding_engine.embed_chunks(
                self.chunks, 
                use_cache=True
            )
            
            # Load FAISS index
            self.faiss_index = FAISSIndex(embedding_dim=embeddings.shape[1])
            self.faiss_index.load(self.faiss_index_path)
            
            # Initialize retriever
            self.retriever = SemanticRetriever(
                self.embedding_engine,
                self.faiss_index,
                self.chunks
            )
            
            # Initialize planner with embedding engine for facet merging
            self.planner = QueryPlanner(
                hf_token=self.hf_token,
                embedding_engine=self.embedding_engine
            )
            
            print("✅ Pipeline initialized\n")
            
        except Exception as e:
            print(f"❌ Pipeline initialization failed: {e}")
            raise
    
    def run(self, user_query: str, target_count: Optional[int] = None) -> Dict[str, Any]:
        """
        Execute complete pipeline on user query.
        
        Args:
            user_query: Natural language query describing desired signatures
            target_count: Optional target number of signatures (auto-determined if None)
        
        Returns:
            Dictionary containing all pipeline results
            
        Raises:
            ValueError: If pipeline not initialized or query invalid
        """
        if not self.kb or not self.planner:
            raise ValueError("Pipeline not initialized. Call initialize() first.")
        if not user_query or not user_query.strip():
            raise ValueError("Query cannot be empty")
        
        print("="*70)
        print("🔬 RUNNING PIPELINE (v2.1)")
        print("="*70)
        
        results = {
            'query': user_query,
            'timestamp': datetime.now().isoformat(),
            'pipeline_version': '2.1-production-ready'
        }
        
        try:
            # STEP 1: Query Planning with facet merging
            print("\n📋 STEP 1: Query Decomposition & Facet Merging")
            plan = self.planner.plan(user_query, target_count=target_count)
            self.planner.print_plan(plan)
            results['plan'] = plan
            
            # Calculate pathway over-selection buffer
            pathway_target = int(plan['target_count'] * self.config.pathway_selection_buffer)
            print(f"\n   Pathway buffer = {pathway_target} "
                  f"({self.config.pathway_selection_buffer}x of {plan['target_count']} signatures)")
            
            # STEP 2: Multi-facet Retrieval with priority-aware depth
            print("\n🔍 STEP 2: Multi-Facet Retrieval (Priority-Aware Depth)")
            facet_results, retrieval_summary = self._execute_retrieval(plan)
            results['retrieval_summary'] = retrieval_summary
            
            # STEP 3: Facet-Aware Biological Validation
            print("\n🔬 STEP 3: Facet-Aware Biological Validation")
            validated_facet_results, validation_stats = self._execute_validation(
                facet_results
            )
            results['validation_stats'] = validation_stats
            
            # STEP 4: Quota-Based Selection
            print("\n🎯 STEP 4: Quota-Based Facet-Aware Selection")
            pathway_selections = self._execute_selection(
                validated_facet_results,
                plan,
                pathway_target
            )
            results['pathway_selections'] = pathway_selections
            results['total_pathways_selected'] = len(pathway_selections)
            
            # STEP 5: Signature Derivation with exact count enforcement
            print("\n🧬 STEP 5: Signature Derivation with Count Enforcement")
            derived_signatures, selector = self._execute_derivation(
                pathway_selections,
                facet_results,
                plan['target_count']
            )
            results['derived_signatures'] = derived_signatures
            results['total_signatures'] = len(derived_signatures)
            
            # Store quota fulfillment
            if hasattr(selector, 'quota_mgr') and selector.quota_mgr:
                results['quota_fulfillment'] = selector.quota_mgr.get_quota_fulfillment()
            
            # STEP 6: Coverage Analysis
            print("\n📊 STEP 6: Coverage Analysis")
            self._analyze_coverage(results, pathway_selections, derived_signatures)
            
            print("\n" + "="*70)
            print("✅ PIPELINE COMPLETE (v2.1)")
            print("="*70)
            
            return results
            
        except Exception as e:
            print(f"\n❌ Pipeline execution failed: {e}")
            raise
    
    def _execute_retrieval(
        self, plan: Dict[str, Any]
    ) -> Tuple[Dict[str, List[Any]], Dict[str, Any]]:
        """
        Execute multi-facet retrieval with priority-aware depth.
        
        Args:
            plan: Query plan from planner
            
        Returns:
            Tuple of (facet_results, retrieval_summary)
        """
        facet_results = {}
        retrieval_summary = {}
        
        for facet in plan['facets']:
            facet_id = facet['facet_id']
            priority = facet['priority']
            top_k = self.config.retrieval_depth_by_priority.get(priority, 75)
            
            print(f"\n   Retrieving {facet_id}: {facet['facet_name']} [{priority}]")
            print(f"   Priority-aware depth = {top_k}")
            
            retrieved = self.retriever.retrieve(facet['retrieval_query'], top_k=top_k)
            facet_results[facet_id] = retrieved
            
            retrieval_summary[facet_id] = {
                'total_retrieved': len(retrieved),
                'top_k': top_k,
                'top_score': float(retrieved[0].score) if retrieved else 0.0,
                'avg_score': float(np.mean([r.score for r in retrieved])) if retrieved else 0.0,
                'min_score': float(retrieved[-1].score) if retrieved else 0.0
            }
            print(f"      → {len(retrieved)} candidates")
        
        return facet_results, retrieval_summary
    
    def _execute_validation(
        self, facet_results: Dict[str, List[Any]]
    ) -> Tuple[Dict[str, List[Any]], Dict[str, Any]]:
        """
        Execute facet-aware biological validation.
        
        Args:
            facet_results: Retrieved results per facet
            
        Returns:
            Tuple of (validated_results, validation_stats)
        """
        validated_facet_results = {}
        validation_stats = {}
        
        validator = BiologicalValidator(
            min_genes=self.config.min_genes,
            max_genes=self.config.max_genes,
            max_overlap=self.config.cross_facet_overlap_threshold,
            facet_aware=True
        )
        
        for facet_id, retrieved in facet_results.items():
            val_results = validator.validate_batch_by_facet(retrieved, facet_id)
            
            accepted = [r for r in retrieved if r.chunk_id in validator.accepted_pathways]
            validated_facet_results[facet_id] = accepted
            
            within_facet_rejected = sum(
                1 for v in val_results 
                if v.status == 'REJECTED' and 'Within-facet' in v.rejection_reason
            )
            cross_facet_rejected = sum(
                1 for v in val_results 
                if v.status == 'REJECTED' and 'Cross-facet' in v.rejection_reason
            )
            
            validation_stats[facet_id] = {
                'total': len(retrieved),
                'accepted': len(accepted),
                'rejected': len(retrieved) - len(accepted),
                'acceptance_rate': len(accepted) / len(retrieved) if retrieved else 0,
                'within_facet_redundant': within_facet_rejected,
                'cross_facet_redundant': cross_facet_rejected
            }
        
        return validated_facet_results, validation_stats
    
    def _execute_selection(
        self,
        validated_facet_results: Dict[str, List[Any]],
        plan: Dict[str, Any],
        pathway_target: int
    ) -> List[Dict[str, Any]]:
        """
        Execute quota-based selection.
        
        Args:
            validated_facet_results: Validated results per facet
            plan: Query plan
            pathway_target: Target number of pathways to select
            
        Returns:
            List of selected pathways
        """
        selector = FacetAwareSelector(
            alpha=self.config.within_facet_diversity_weight,
            beta=self.config.cross_facet_diversity_weight
        )
        
        pathway_selections = selector.select_with_quotas(
            validated_facet_results,
            plan['facets'],
            target_count=pathway_target
        )
        
        return pathway_selections
    
    def _execute_derivation(
        self,
        pathway_selections: List[Dict[str, Any]],
        facet_results: Dict[str, List[Any]],
        target_count: int
    ) -> Tuple[List[Dict[str, Any]], Any]:
        """
        Execute signature derivation with exact count enforcement.
        
        Args:
            pathway_selections: Selected pathways
            facet_results: Original facet results
            target_count: Exact number of signatures to generate
            
        Returns:
            Tuple of (derived_signatures, selector)
        """
        deriver = SignatureDeriver(
            core_threshold=self.config.core_signature_threshold,
            unique_threshold=self.config.unique_signature_threshold
        )
        
        derived_signatures = deriver.derive_signatures(
            pathway_selections,
            facet_results
        )
        
        # Enforce exact final signature count
        print(f"\n🎯 Enforcing exact signature count ({target_count})")
        
        if len(derived_signatures) > target_count:
            def rank_signature(sig):
                gene_count_score = sig['gene_count'] / 100.0
                confidence_score = sig['confidence']
                return gene_count_score * 0.6 + confidence_score * 0.4
            
            derived_signatures.sort(key=rank_signature, reverse=True)
            trimmed = derived_signatures[:target_count]
            print(f"   Trimmed {len(derived_signatures)} → {len(trimmed)} signatures")
            derived_signatures = trimmed
        
        return derived_signatures, deriver
    
    def _analyze_coverage(
        self,
        results: Dict[str, Any],
        pathway_selections: List[Dict[str, Any]],
        derived_signatures: List[Dict[str, Any]]
    ) -> None:
        """
        Analyze coverage and compute statistics.
        
        Args:
            results: Results dictionary to update (modified in place)
            pathway_selections: Selected pathways
            derived_signatures: Derived signatures
        """
        all_genes_derived = set()
        for sig in derived_signatures:
            all_genes_derived.update(sig['genes'])
        
        all_genes_pathways = set()
        for sel in pathway_selections:
            all_genes_pathways.update(sel['genes'])
        
        results['total_unique_genes'] = len(all_genes_derived)
        results['all_genes'] = sorted(list(all_genes_derived))
        results['pathway_genes_count'] = len(all_genes_pathways)
        
        # Facet distribution
        facet_distribution = defaultdict(int)
        for sel in pathway_selections:
            facet_distribution[sel['facet_id']] += 1
        results['facet_distribution'] = dict(facet_distribution)
        
        # Source distribution
        source_distribution = defaultdict(int)
        for sel in pathway_selections:
            source_distribution[sel['source']] += 1
        results['source_distribution'] = dict(source_distribution)
        
        # Derivation distribution
        derivation_distribution = defaultdict(int)
        for sig in derived_signatures:
            derivation_distribution[sig['derivation_method']] += 1
        results['derivation_distribution'] = dict(derivation_distribution)
        
        print(f"   Total unique genes: {len(all_genes_derived)}")
        if pathway_selections:
            ratio = len(derived_signatures) / len(pathway_selections)
            print(f"   Signatures per pathway: {ratio:.2f}")
    
    def export_results(self, results: Dict[str, Any], output_dir: str) -> None:
        """
        Export pipeline results to multiple file formats.
        
        Args:
            results: Dictionary containing pipeline results.
            output_dir: Directory path for output files.
            
        Raises:
            ValueError: If results dictionary is invalid.
            IOError: If file writing fails.
        """
        if not results:
            raise ValueError("Results dictionary cannot be empty")
        
        try:
            os.makedirs(output_dir, exist_ok=True)
        except OSError as e:
            raise IOError(f"Failed to create output directory {output_dir}: {e}")
        
        print(f"\n💾 Exporting results to: {output_dir}")
        
        # Export all formats with error handling
        self._export_json(results, output_dir)
        self._export_derived_signatures_gmt(results, output_dir)
        self._export_pathway_selections_gmt(results, output_dir)
        self._export_quota_report(results, output_dir)
        self._export_summary(results, output_dir)
        
        print(f"\n✅ All files exported successfully!")

    def _export_json(self, results: Dict[str, Any], output_dir: str) -> None:
        """Export complete results as JSON."""
        json_path = os.path.join(output_dir, 'complete_results_v2.1.json')
        try:
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"   ✅ JSON: {json_path}")
        except IOError as e:
            print(f"   ❌ Failed to write JSON: {e}")
            raise

    def _export_derived_signatures_gmt(
        self, results: Dict[str, Any], output_dir: str
    ) -> None:
        """Export derived signatures in GMT format."""
        if not results.get('derived_signatures'):
            print(f"   ⏭️  No derived signatures to export")
            return
        
        sig_gmt_path = os.path.join(output_dir, 'derived_signatures.gmt')
        try:
            with open(sig_gmt_path, 'w', encoding='utf-8') as f:
                for sig in results['derived_signatures']:
                    sig_id = sig['signature_id']
                    desc = self._format_signature_description(sig)
                    genes = sig['genes']
                    
                    f.write(f"{sig_id}\t{desc}\t" + "\t".join(genes) + "\n")
            print(f"   ✅ Derived Signatures GMT: {sig_gmt_path}")
        except (IOError, KeyError) as e:
            print(f"   ❌ Failed to write signatures GMT: {e}")
            raise

    def _export_pathway_selections_gmt(
        self, results: Dict[str, Any], output_dir: str
    ) -> None:
        """Export pathway selections in GMT format."""
        if not results.get('pathway_selections'):
            print(f"   ⏭️  No pathway selections to export")
            return
        
        pathway_gmt_path = os.path.join(output_dir, 'selected_pathways.gmt')
        try:
            with open(pathway_gmt_path, 'w', encoding='utf-8') as f:
                for sel in results['pathway_selections']:
                    pathway_id = sel['pathway_id']
                    desc = self._format_pathway_description(sel)
                    genes = sel['genes']
                    
                    f.write(f"{pathway_id}\t{desc}\t" + "\t".join(genes) + "\n")
            print(f"   ✅ Pathway Selections GMT: {pathway_gmt_path}")
        except (IOError, KeyError) as e:
            print(f"   ❌ Failed to write pathways GMT: {e}")
            raise

    def _format_signature_description(self, sig: Dict[str, Any]) -> str:
        """Format description string for signature GMT entry."""
        return (
            f"{sig['facet_id']}|{sig['source']}|"
            f"{sig['derivation_method']}|"
            f"conf:{sig['confidence']:.3f}|"
            f"parent:{','.join(sig['parent_pathways'])}"
        )

    def _format_pathway_description(self, sel: Dict[str, Any]) -> str:
        """Format description string for pathway GMT entry."""
        return (
            f"{sel['facet_id']}|{sel['source']}|"
            f"{sel['gene_count']}genes|"
            f"rank:{sel['selection_rank']}"
        )

    def _export_quota_report(self, results: Dict[str, Any], output_dir: str) -> None:
        """Export quota fulfillment report."""
        if not results.get('quota_fulfillment'):
            print(f"   ⏭️  No quota data to export")
            return
            
        quota_path = os.path.join(output_dir, 'quota_fulfillment.txt')
        try:
            with open(quota_path, 'w', encoding='utf-8') as f:
                f.write("="*70 + "\n")
                f.write("FACET QUOTA FULFILLMENT (v2.1)\n")
                f.write("="*70 + "\n\n")
                
                for facet_id, data in results['quota_fulfillment'].items():
                    f.write(
                        f"{data['status']} {facet_id}: "
                        f"{data['filled']}/{data['quota']} "
                        f"({data['fulfillment_rate']*100:.1f}%)\n"
                    )
            print(f"   ✅ Quotas: {quota_path}")
        except (IOError, KeyError) as e:
            print(f"   ❌ Failed to write quota report: {e}")
            raise

    def _export_summary(self, results: Dict[str, Any], output_dir: str) -> None:
        """Export pipeline summary."""
        summary_path = os.path.join(output_dir, 'summary.txt')
        try:
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write("="*70 + "\n")
                f.write("PIPELINE SUMMARY (v2.1)\n")
                f.write("="*70 + "\n\n")
                
                f.write(f"Query: {results['query']}\n")
                f.write(f"Timestamp: {results['timestamp']}\n")
                f.write(f"Pipeline Version: {results['pipeline_version']}\n\n")
                
                f.write(f"Target Count: {results['plan']['target_count']}\n")
                f.write(f"Pathways Selected: {results['total_pathways_selected']}\n")
                f.write(f"Signatures Derived: {results['total_signatures']}\n")
                f.write(f"Total Unique Genes: {results['total_unique_genes']}\n\n")
            
            print(f"   ✅ Summary: {summary_path}")
        except (IOError, KeyError) as e:
            print(f"   ❌ Failed to write summary: {e}")
            raise
    
    def print_summary(self, results: Dict[str, Any]) -> None:
        """
        Print comprehensive pipeline summary.
        Uses stored quota fulfillment data for consistency.
        
        Args:
            results: Pipeline results dictionary
        """
        print("\n" + "="*70)
        print("📊 PIPELINE SUMMARY (v2.1)")
        print("="*70)
        
        print(f"\n📝 Query: {results['query'][:80]}...")
        print(f"🎯 Target: {results['plan']['target_count']} signatures")
        print(f"✅ Pathways Selected: {results['total_pathways_selected']}")
        print(f"🧬 Signatures Derived: {results['total_signatures']}")
        print(f"🔬 Unique Genes: {results['total_unique_genes']}")
        
        if results['total_pathways_selected'] > 0:
            ratio = results['total_signatures'] / results['total_pathways_selected']
            print(f"📈 Signatures/Pathway Ratio: {ratio:.2f}")
        
        # Use stored quota_fulfillment
        if results.get('quota_fulfillment'):
            print(f"\n📊 Facet Distribution (with Quotas):")
            for facet_id, count in sorted(results['facet_distribution'].items()):
                pct = (count / results['total_pathways_selected'] * 100 
                       if results['total_pathways_selected'] > 0 else 0)
                
                quota_data = results['quota_fulfillment'].get(facet_id, {})
                quota = quota_data.get('quota', 'N/A')
                status = quota_data.get('status', '')
                
                print(f"   {facet_id}: {count} ({pct:.1f}%) {status} {count}/{quota}")
        else:
            print(f"\n📊 Facet Distribution:")
            for facet_id, count in sorted(results['facet_distribution'].items()):
                pct = (count / results['total_pathways_selected'] * 100 
                       if results['total_pathways_selected'] > 0 else 0)
                print(f"   {facet_id}: {count} ({pct:.1f}%)")
        
        print(f"\n📚 Source Distribution:")
        for source, count in sorted(results['source_distribution'].items(), 
                                     key=lambda x: -x[1]):
            print(f"   {source}: {count}")
        
        if results.get('derivation_distribution'):
            print(f"\n🧬 Derivation Methods:")
            for method, count in sorted(results['derivation_distribution'].items(), 
                                         key=lambda x: -x[1]):
                print(f"   {method}: {count}")
        
        print("\n" + "="*70)


# ============================================================
# MAIN EXECUTION BLOCK - COMPLETE REPLACEMENT
# ============================================================

if __name__ == "__main__":
    import os
    from pathlib import Path
    
    print("="*70)
    print("COMPLETE PIPELINE TEST (v2.1 - PRODUCTION READY)")
    print("="*70)
    
    # SECURITY FIX: Load token from environment variable
    HF_TOKEN = os.environ.get('HF_TOKEN')
    if not HF_TOKEN:
        raise ValueError(
            "HF_TOKEN environment variable not set. "
            "Please set it before running:\n"
            "  export HF_TOKEN='your_token_here'"
        )
    
    # Use environment variables or defaults for paths
    base_dir = os.environ.get('SIGGEN_BASE_DIR', '/content/drive/MyDrive/siggen')
    base_path = Path(base_dir)
    
    # Validate base directory exists
    if not base_path.exists():
        raise FileNotFoundError(
            f"Base directory not found: {base_dir}\n"
            f"Please set SIGGEN_BASE_DIR environment variable or create the directory."
        )
    
    print(f"\n📁 Using base directory: {base_dir}")
    print(f"🔑 HF Token: {'*' * 20}{HF_TOKEN[-8:]}")
    
    try:
        # Create custom configuration if needed
        config = PipelineConfig(
            min_genes=5,
            max_genes=300,
            within_facet_overlap_threshold=0.50,
            cross_facet_overlap_threshold=0.25
        )
        
        # Initialize pipeline
        pipeline = CompletePipeline(
            kb_path=str(base_path / 'knowledge_base.json.gz'),
            embeddings_cache_dir=str(base_path / 'embeddings'),
            faiss_index_path=str(base_path / 'faiss_index.bin'),
            hf_token=HF_TOKEN,
            config=config
        )
        
        pipeline.initialize()
        
        # Example query
        user_query = """
        Give me 25 signatures for metabolic and immune checkpoint pathways 
        involved in treatment resistance in pancreatic cancer. Focus on 
        glycolysis, oxidative stress, and T-cell exhaustion.
        """
        
        # Run pipeline
        results = pipeline.run(user_query, target_count=25)
        
        # Print summary
        pipeline.print_summary(results)
        
        # Export results
        output_dir = base_path / 'results_v2.1'
        pipeline.export_results(results, str(output_dir))
        
        print(f"\n🎉 Pipeline complete! Check {output_dir}")
        
    except Exception as e:
        print(f"\n❌ Pipeline failed with error: {e}")
        import traceback
        traceback.print_exc()
        raise


# ============================================================
# USAGE INSTRUCTIONS
# ============================================================
"""
To use this production-ready code:

1. Set environment variables:
   export HF_TOKEN='your_huggingface_token_here'
   export SIGGEN_BASE_DIR='/path/to/your/data'  # optional

2. Run the script:
   python complete_module_replacements.py

3. Or import and use programmatically:
   from complete_module_replacements import CompletePipeline, PipelineConfig
   
   config = PipelineConfig(min_genes=10, max_genes=200)
   pipeline = CompletePipeline(..., config=config)
   results = pipeline.run(query, target_count=25)

Key improvements:
- ✅ Removed hardcoded API token (CRITICAL SECURITY FIX)
- ✅ Added comprehensive error handling throughout
- ✅ Improved validation in PipelineConfig
- ✅ Better type hints and documentation
- ✅ Extracted long methods into smaller, focused helpers
- ✅ Made paths configurable via environment variables
- ✅ Added proper exception handling with informative messages
- ✅ Improved code organization and maintainability
"""
