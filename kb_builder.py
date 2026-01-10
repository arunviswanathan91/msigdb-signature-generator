"""
Knowledge Base Builder Module
==============================
Build custom knowledge bases from GMT files with real-time progress tracking.
"""

import json
import gzip
import os
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Tuple, Any
import tempfile


class KBBuilder:
    """Build knowledge base from uploaded GMT files"""
    
    def __init__(self, min_genes: int = 5, max_genes: int = 500):
        """
        Initialize KB Builder
        
        Args:
            min_genes: Minimum genes per pathway
            max_genes: Maximum genes per pathway
        """
        self.min_genes = min_genes
        self.max_genes = max_genes
        self.stats = defaultdict(lambda: {'total': 0, 'kept': 0, 'too_small': 0, 
                                          'too_large': 0, 'duplicates': 0})
    
    def parse_gmt_file(self, file_content: str, source_name: str) -> Dict[str, List[str]]:
        """
        Parse a single GMT file
        
        Args:
            file_content: GMT file content as string
            source_name: Name/identifier for this source
            
        Returns:
            Dictionary of pathway_name -> gene_list
        """
        pathways = {}
        
        for line in file_content.strip().split('\n'):
            if not line.strip():
                continue
                
            parts = line.strip().split('\t')
            
            if len(parts) < 3:
                continue
            
            pathway_name = parts[0]
            genes = [g.strip() for g in parts[2:] if g.strip()]
            
            self.stats[source_name]['total'] += 1
            
            # Apply filters
            if len(genes) < self.min_genes:
                self.stats[source_name]['too_small'] += 1
                continue
            
            if len(genes) > self.max_genes:
                self.stats[source_name]['too_large'] += 1
                continue
            
            # Handle duplicates
            if pathway_name in pathways:
                self.stats[source_name]['duplicates'] += 1
                # Keep the one with more genes
                if len(genes) > len(pathways[pathway_name]):
                    pathways[pathway_name] = genes
            else:
                pathways[pathway_name] = genes
                self.stats[source_name]['kept'] += 1
        
        return pathways
    
    def build_kb(self, uploaded_files: List[Tuple[str, str]]) -> Dict[str, Any]:
        """
        Build complete knowledge base from multiple GMT files
        
        Args:
            uploaded_files: List of (filename, content) tuples
            
        Returns:
            Complete KB with metadata
        """
        knowledge_base = {}
        
        # Process each file
        for filename, content in uploaded_files:
            # Extract friendly name
            source_name = self._extract_source_name(filename)
            
            # Parse GMT
            pathways = self.parse_gmt_file(content, source_name)
            
            # Merge into main KB (pathways dict already handles duplicates)
            knowledge_base.update(pathways)
        
        # Calculate statistics
        total_genes = set()
        for genes in knowledge_base.values():
            total_genes.update(genes)
        
        # Create complete KB structure
        kb_with_metadata = {
            'metadata': {
                'created': datetime.now().isoformat(),
                'version': '2025.1',
                'total_pathways': len(knowledge_base),
                'total_unique_genes': len(total_genes),
                'sources': {name: self.stats[name]['kept'] for name in self.stats.keys()},
                'filters': {
                    'min_genes': self.min_genes,
                    'max_genes': self.max_genes
                },
                'stats': dict(self.stats)
            },
            'pathways': knowledge_base
        }
        
        return kb_with_metadata
    
    def save_kb(self, kb_data: Dict[str, Any], output_path: str) -> int:
        """
        Save knowledge base to compressed JSON
        
        Args:
            kb_data: Complete KB with metadata
            output_path: Path to save .json.gz file
            
        Returns:
            File size in bytes
        """
        with gzip.open(output_path, 'wt', encoding='UTF-8') as zipfile:
            json.dump(kb_data, zipfile, indent=2)
        
        return os.path.getsize(output_path)
    
    def _extract_source_name(self, filename: str) -> str:
        """Extract friendly source name from filename"""
        filename = filename.lower()
        
        if 'kegg' in filename:
            return 'KEGG'
        elif 'reactome' in filename:
            return 'Reactome'
        elif 'wikipathways' in filename:
            return 'WikiPathways'
        elif 'immport' in filename:
            return 'ImmPort'
        elif 'hallmark' in filename or 'h.all' in filename:
            return 'Hallmark'
        elif 'oncogenic' in filename or 'c6' in filename:
            return 'Oncogenic'
        elif 'gene_ontology' in filename or 'c5' in filename:
            return 'Gene_Ontology'
        elif 'biocarta' in filename:
            return 'BioCarta'
        elif 'pid' in filename:
            return 'PID'
        else:
            # Use filename without extension
            return filename.replace('.gmt', '').replace('.txt', '').upper()
    
    def get_summary(self) -> Dict[str, Any]:
        """Get build summary statistics"""
        return {
            'sources': dict(self.stats),
            'total_sources': len(self.stats),
            'total_pathways_kept': sum(s['kept'] for s in self.stats.values()),
            'total_pathways_filtered': sum(s['too_small'] + s['too_large'] 
                                          for s in self.stats.values())
        }


def validate_gmt_content(content: str) -> Tuple[bool, str]:
    """
    Validate GMT file format
    
    Args:
        content: File content as string
        
    Returns:
        (is_valid, error_message)
    """
    lines = [l.strip() for l in content.strip().split('\n') if l.strip()]
    
    if not lines:
        return False, "File is empty"
    
    # Check first few lines for GMT format
    valid_lines = 0
    for i, line in enumerate(lines[:10]):
        parts = line.split('\t')
        if len(parts) >= 3:
            valid_lines += 1
    
    if valid_lines < len(lines[:10]) * 0.8:  # At least 80% should be valid
        return False, "File does not appear to be in GMT format (tab-separated: pathway_name, description, genes...)"
    
    return True, ""
