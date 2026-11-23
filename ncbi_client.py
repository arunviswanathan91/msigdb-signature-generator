"""
NCBI Gene client: gene aliases -> official symbols
==================================================

LLMs often name genes by a familiar alias (GLUT1, PD-L1, 4-1BB, NRF2) even
when asked for official symbols, and the knowledge base only knows official
ones, so those seeds used to be silently lost. This module asks NCBI Gene,
through the public E-utilities API, which official human gene(s) carry a given
name.

The approach follows GeneGPT (Jin et al., Bioinformatics 2024), which showed
that grounding gene nomenclature in NCBI's Web APIs beats a language model's
memory. Unlike GeneGPT, no model composes the queries here: they are built
deterministically, so the same input always asks the same question.

Two requests resolve any number of names: one ``esearch`` over the Gene Name
field (official symbols and aliases), one ``esummary`` for the hits. A name is
accepted only on an exact match (ignoring case and punctuation) with a gene's
official symbol or one of its listed aliases.

NCBI asks clients to stay under 3 requests/second without an API key (10 with
one, read from ``NCBI_API_KEY``). No personal data is sent.
"""

from __future__ import annotations

import os
import re
import time
from typing import Dict, List, Optional, Sequence

import requests

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
TOOL_NAME = "msigdb-signature-generator"
_SAFE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 ._/\-]{0,40}$")


class NCBIUnavailable(ConnectionError):
    """NCBI E-utilities could not be reached or kept returning errors."""


def normalise_name(name: str) -> str:
    """Case- and punctuation-insensitive key: 'PD-L1', 'pdl1' and 'PD L1' are the same."""
    return "".join(ch for ch in name.upper() if ch.isalnum())


class NCBIGene:
    """Minimal, polite client for the NCBI Gene database."""

    def __init__(self, session: Optional[requests.Session] = None, api_key: Optional[str] = None,
                 timeout: float = 15.0):
        self.session = session or requests.Session()
        self.api_key = api_key if api_key is not None else os.environ.get("NCBI_API_KEY", "").strip()
        self.timeout = timeout
        self.min_interval = 0.11 if self.api_key else 0.34
        self._last_request = 0.0

    def _get(self, endpoint: str, params: Dict) -> Dict:
        params = {**params, "retmode": "json", "tool": TOOL_NAME}
        if self.api_key:
            params["api_key"] = self.api_key
        last_error: Optional[str] = None
        for attempt in range(3):
            wait = self.min_interval - (time.monotonic() - self._last_request)
            if wait > 0:
                time.sleep(wait)
            self._last_request = time.monotonic()
            try:
                response = self.session.get(f"{EUTILS}/{endpoint}", params=params, timeout=self.timeout)
            except requests.RequestException as exc:
                last_error = str(exc)
            else:
                if response.status_code == 200:
                    try:
                        return response.json()
                    except ValueError:
                        last_error = "NCBI returned a non-JSON response"
                elif response.status_code in (429, 500, 502, 503, 504):
                    last_error = f"NCBI returned HTTP {response.status_code}"
                else:
                    raise NCBIUnavailable(f"NCBI returned HTTP {response.status_code}")
            time.sleep(1.0 * (attempt + 1))
        raise NCBIUnavailable(f"NCBI E-utilities unavailable: {last_error}")

    def lookup(self, names: Sequence[str], organism: str = "Homo sapiens") -> Dict[str, List[Dict]]:
        """
        Official genes for each name.

        Returns {name: [{"symbol", "gene_id", "description", "match"}]} where
        ``match`` is "official" when the name is itself an official symbol and
        "alias" otherwise. A name with no exact match maps to []. More than one
        entry means the name is ambiguous (AMPK, NRF2, PD1).

        Raises NCBIUnavailable if the API cannot be reached.
        """
        wanted = [n.strip() for n in dict.fromkeys(names) if isinstance(n, str) and _SAFE.match(n.strip())]
        results: Dict[str, List[Dict]] = {n: [] for n in names if isinstance(n, str)}
        if not wanted:
            return results

        for start in range(0, len(wanted), 40):
            chunk = wanted[start:start + 40]
            term = "(" + " OR ".join(f'"{name}"[gene]' for name in chunk) + \
                   f') AND "{organism}"[Organism] AND alive[prop]'
            found = self._get("esearch.fcgi", {"db": "gene", "term": term, "retmax": 500,
                                               "sort": "relevance"})
            ids = found.get("esearchresult", {}).get("idlist", [])
            documents: List[Dict] = []
            for offset in range(0, len(ids), 200):
                summary = self._get("esummary.fcgi", {"db": "gene", "id": ",".join(ids[offset:offset + 200])})
                result = summary.get("result", {})
                documents.extend(result[uid] for uid in result.get("uids", []) if uid in result)

            keys = {name: normalise_name(name) for name in chunk}
            for document in documents:
                symbol = document.get("name")
                if not symbol:
                    continue
                official = normalise_name(symbol)
                aliases = {normalise_name(a) for a in document.get("otheraliases", "").split(",") if a.strip()}
                for name, key in keys.items():
                    if key == official or key in aliases:
                        results[name].append({
                            "symbol": symbol,
                            "gene_id": document.get("uid"),
                            "description": document.get("description", ""),
                            "match": "official" if key == official else "alias",
                        })

        for name, hits in results.items():
            # An exact official-symbol match settles it, whatever else lists the name as an alias.
            official = [h for h in hits if h["match"] == "official"]
            results[name] = official[:1] if official else hits
        return results
