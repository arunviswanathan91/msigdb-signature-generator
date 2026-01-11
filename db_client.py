"""
Database Client - For Streamlit App
Makes HTTP requests to remote database server
No local database files needed!
"""

import requests
import streamlit as st
from typing import List, Dict, Set
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

class DatabaseClient:
    """
    Client for remote database API
    User's Streamlit app uses this instead of loading local .db files
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
            st.sidebar.success("🟢 Database connected")
        except Exception as e:
            st.sidebar.error(f"🔴 Database offline: {e}")
            st.error("Cannot connect to database server. Please check API URL.")
            st.stop()
    
    
    def expand_signature_smart(
        self,
        seed_genes: List[str],
        strength: float = 0.5,
        max_pathways_per_gene: int = 5,
        min_pathway_prob: float = 0.05,
        min_gene_prob: float = 0.05
    ) -> Set[str]:
        """
        Smart signature expansion using probabilistic networks
        
        This is the MAIN method - all the heavy computation happens on the server!
        """
        # Build query parameters
        params = {
            'strength': strength,
            'max_pathways_per_gene': max_pathways_per_gene,
            'min_pathway_prob': min_pathway_prob,
            'min_gene_prob': min_gene_prob
        }
        
        # Send genes as a LIST in the body (not wrapped in object!)
        response = self.session.post(
            f"{self.api_url}/api/expand-signature",
            json=seed_genes,  # ← JUST THE LIST, NOT A DICT!
            params=params,    # ← Other params as query params
            timeout=30
        )
        response.raise_for_status()
        result = response.json()
        return set(result['expanded_genes'])
    
    def get_stats(self) -> Dict:
        """Get database statistics"""
        response = self.session.get(
            f"{self.api_url}/api/stats",
            timeout=10
        )
        response.raise_for_status()
        return response.json()
