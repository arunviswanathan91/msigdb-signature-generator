"""
Pathway API client
==================

Single client for the MSigDB API Space (replaces the former db_client.py /
db_client_enhanced.py pair; the remote search cache is gone).

Design rules, each of which fixes a concrete earlier fault:

* Thresholds are on the scale of the data. Gene-gene "probabilities" are
  row-normalised transition weights whose maximum is ~0.05-0.07, so the old
  cut-offs of 0.05-0.10 returned nothing for most genes.
* Evidence is measured *relative to the signature* (does this gene link to the
  other genes here?) rather than by global degree, which flagged exactly the
  genes with the strongest partners.
* A failed request is reported as "unavailable". It is never converted into
  negative evidence about a gene.
* "Not covered by the evidence table" (it holds 10,000 genes) is reported
  separately from "weak evidence". ``literature_count`` is not used: the
  server caps it at 5 for every gene.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Sequence

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from gene_filters import is_housekeeping

NEIGHBOUR_TOP_K = 50          # server maximum
NEIGHBOUR_FLOOR = 0.01        # weakest link considered at all
LOW_EXPRESSION_TPM = 1.0


class BackendUnavailable(ConnectionError):
    """The pathway API could not be reached or returned a server error."""


def strength_to_min_prob(strength: float) -> float:
    """
    Map expansion strength in [0, 1] to a neighbour-probability cut-off.

    Higher strength = more permissive = more genes. 0 -> 0.05 (only the
    strongest links), 1 -> 0.01 (every link the server reports).
    """
    strength = min(1.0, max(0.0, float(strength)))
    return round(0.05 - 0.04 * strength, 4)


class DatabaseClient:
    """REST client for the pathway API."""

    def __init__(self, api_url: str, timeout: float = 15.0, workers: int = 8):
        self.api_url = api_url.rstrip('/')
        self.timeout = timeout
        self.workers = workers
        self.session = requests.Session()
        retry = Retry(total=3, backoff_factor=0.4, status_forcelist=[500, 502, 503, 504],
                      allowed_methods=frozenset(["GET", "POST"]))
        adapter = HTTPAdapter(max_retries=retry, pool_maxsize=workers * 2)
        self.session.mount('http://', adapter)
        self.session.mount('https://', adapter)
        self._neighbour_cache: Dict[str, List[Dict]] = {}

    # ---------------------------------------------------------------- transport

    def _request(self, method: str, endpoint: str, **kwargs):
        try:
            response = self.session.request(
                method, f"{self.api_url}{endpoint}", timeout=self.timeout, **kwargs)
        except requests.RequestException as exc:
            raise BackendUnavailable(f"Pathway API unreachable: {exc}") from exc
        if response.status_code >= 500:
            raise BackendUnavailable(f"Pathway API error {response.status_code} on {endpoint}")
        return response

    def _json(self, method: str, endpoint: str, **kwargs):
        response = self._request(method, endpoint, **kwargs)
        response.raise_for_status()
        return response.json()

    # ---------------------------------------------------------------- endpoints

    def health_check(self) -> Dict:
        return self._json("GET", "/")

    def get_stats(self) -> Dict:
        return self._json("GET", "/api/stats")

    def get_gene_neighbors(self, gene: str, min_prob: float = NEIGHBOUR_FLOOR,
                           top_k: int = NEIGHBOUR_TOP_K) -> List[Dict]:
        """Neighbours of ``gene`` as [{target, probability, rank}], strongest first."""
        if gene not in self._neighbour_cache:
            self._neighbour_cache[gene] = self._json(
                "GET", f"/api/gene-neighbors/{requests.utils.quote(gene, safe='')}",
                params={"min_prob": NEIGHBOUR_FLOOR, "top_k": NEIGHBOUR_TOP_K})
        return [n for n in self._neighbour_cache[gene] if n["probability"] >= min_prob][:top_k]

    def neighbours_for(self, genes: Sequence[str]) -> Dict[str, List[Dict]]:
        """Neighbour lists for many genes, fetched concurrently."""
        missing = [g for g in dict.fromkeys(genes) if g not in self._neighbour_cache]
        if missing:
            with ThreadPoolExecutor(max_workers=self.workers) as pool:
                list(pool.map(self.get_gene_neighbors, missing))
        return {g: self._neighbour_cache[g] for g in genes}

    def get_expression_batch(self, genes: Sequence[str]) -> Dict[str, Dict[str, float]]:
        """{GENE: {tissue: median TPM}} for genes present in GTEx v10."""
        return self._json("POST", "/api/expression/batch", json=list(genes)).get("results", {})

    def get_evidence_batch(self, genes: Sequence[str]) -> Dict[str, Dict]:
        """{GENE: evidence record} for genes present in the evidence table."""
        return self._json("POST", "/api/evidence/batch", json=list(genes)).get("results", {})

    # ---------------------------------------------------------------- expansion

    def network_expansion(
        self,
        seed_genes: Sequence[str],
        strength: float = 0.5,
        max_additions: int = 5,
        min_supporters: int = 2,
        drop_housekeeping: bool = True,
    ) -> List[Dict]:
        """
        Propose genes to add from the gene-gene network.

        A candidate must be a neighbour of at least ``min_supporters`` different
        signature genes - convergence on the signature, not a single strong
        partner. Candidates are ranked by the summed link probability.

        Returns [{gene, score, supporters: [seed, ...]}], best first, at most
        ``max_additions`` long. Never returns a gene already in the signature.
        """
        min_prob = strength_to_min_prob(strength)
        seeds = list(dict.fromkeys(seed_genes))
        present = {g.upper() for g in seeds}
        neighbours = self.neighbours_for(seeds)

        score: Dict[str, float] = {}
        supporters: Dict[str, List[str]] = {}
        for seed in seeds:
            for link in neighbours[seed]:
                target = link["target"]
                if link["probability"] < min_prob or target.upper() in present:
                    continue
                score[target] = score.get(target, 0.0) + link["probability"]
                supporters.setdefault(target, []).append(seed)

        candidates = [
            {"gene": g, "score": round(score[g], 5), "supporters": supporters[g]}
            for g in score
            if len(supporters[g]) >= min_supporters
            and not (drop_housekeeping and is_housekeeping(g))
        ]
        candidates.sort(key=lambda c: (-c["score"], c["gene"]))
        return candidates[:max_additions]

    # ---------------------------------------------------------------- evidence

    def signature_evidence(self, genes: Sequence[str], tissue_context: Optional[str] = None) -> Dict:
        """
        Per-gene database evidence for one signature.

        Returns:
            {
              "genes": {GENE: {
                  "links_in_signature": int,      # other signature genes it is linked to
                  "linked_to": [GENE, ...],
                  "tissue_tpm": float | None,     # None = no tissue given or gene not in GTEx
                  "pathway_count": int | None,    # None = not covered by the evidence table
                  "evidence_strength": str | None,
                  "housekeeping": bool,
                  "flags": [str, ...],            # isolated / low_expression / housekeeping
              }},
              "tissue": matched GTEx tissue key or None,
              "tissue_requested": str | None,
              "cohesion": fraction of genes with >= 1 link inside the signature,
              "sources": [str, ...],
            }

        Raises BackendUnavailable if the API cannot be reached.
        """
        genes = list(dict.fromkeys(genes))
        upper = {g.upper(): g for g in genes}

        neighbours = self.neighbours_for(genes)
        links: Dict[str, set] = {g: set() for g in genes}
        for gene in genes:
            for link in neighbours[gene]:
                partner = upper.get(link["target"].upper())
                if partner and partner != gene:
                    links[gene].add(partner)
                    links[partner].add(gene)      # the network is treated as undirected

        evidence = self.get_evidence_batch(genes)
        sources = ["gene_gene.db", "gene_evidence_v1.pkl.gz"]

        expression: Dict[str, Dict[str, float]] = {}
        tissue_key = None
        if tissue_context:
            expression = self.get_expression_batch(genes)
            sources.append("gtex_expression_v10.pkl.gz")
            available = next(iter(expression.values()), {})
            tissue_key = match_tissue(tissue_context, list(available))

        table: Dict[str, Dict] = {}
        for gene in genes:
            record = evidence.get(gene.upper())
            tissues = expression.get(gene.upper())
            tpm = tissues.get(tissue_key) if (tissues and tissue_key) else None
            housekeeping = is_housekeeping(gene)

            flags = []
            if not links[gene]:
                flags.append("isolated")
            if tpm is not None and tpm < LOW_EXPRESSION_TPM:
                flags.append("low_expression")
            if housekeeping:
                flags.append("housekeeping")

            table[gene] = {
                "links_in_signature": len(links[gene]),
                "linked_to": sorted(links[gene]),
                "tissue_tpm": None if tpm is None else round(float(tpm), 2),
                "pathway_count": record.get("pathway_count") if record else None,
                "evidence_strength": record.get("evidence_strength") if record else None,
                "housekeeping": housekeeping,
                "flags": flags,
            }

        linked = sum(1 for g in genes if links[g])
        return {
            "genes": table,
            "tissue": tissue_key,
            "tissue_requested": tissue_context,
            "cohesion": round(linked / len(genes), 4) if genes else 0.0,
            "sources": sources,
        }


def match_tissue(query: str, available: Sequence[str]) -> Optional[str]:
    """Match a user tissue label to a GTEx tissue key. None if nothing fits."""
    if not query or query.strip().lower() in ("unspecified", "custom"):
        return None
    wanted = query.strip().lower()

    for tissue in available:
        if tissue.lower() == wanted:
            return tissue
    for tissue in available:
        if wanted in tissue.lower().replace('_', ' ') or tissue.lower().replace('_', ' ') in wanted:
            return tissue

    aliases = {
        'pbmc': 'blood', 'fat': 'adipose', 'colon': 'colon', 'gut': 'intestine',
        'cortex': 'brain', 'hepat': 'liver', 'renal': 'kidney', 'cardiac': 'heart',
        'skeletal': 'muscle', 'derm': 'skin', 'pulmon': 'lung', 'islet': 'pancreas',
    }
    for alias, canonical in aliases.items():
        if alias in wanted:
            for tissue in available:
                if canonical in tissue.lower():
                    return tissue
    return None


def format_evidence_table(evidence: Dict) -> str:
    """Render ``signature_evidence`` output as a compact table for an LLM prompt."""
    tissue = evidence.get("tissue")
    header = ["gene", "links_to_signature", "kb_gene_sets"]
    if tissue:
        header.append(f"TPM_{tissue}")
    header.append("flags")

    lines = [" | ".join(header)]
    for gene, row in evidence["genes"].items():
        cells = [
            gene,
            str(row["links_in_signature"]),
            "not_covered" if row["pathway_count"] is None else str(row["pathway_count"]),
        ]
        if tissue:
            cells.append("n/a" if row["tissue_tpm"] is None else f"{row['tissue_tpm']:g}")
        cells.append(",".join(row["flags"]) or "-")
        lines.append(" | ".join(cells))

    notes = [
        "",
        "links_to_signature = number of OTHER genes in this signature the gene is linked to "
        "in the gene-gene co-membership network (0 = isolated from the rest).",
        "kb_gene_sets = number of knowledge-base gene sets containing the gene; "
        "'not_covered' means the evidence table has no record, NOT that evidence is weak.",
    ]
    if tissue:
        notes.append(f"TPM = GTEx v10 median expression in {tissue}; "
                     f"low_expression means < {LOW_EXPRESSION_TPM:g} TPM. "
                     "Bulk tissue TPM can miss genes restricted to rare cell types.")
    elif evidence.get("tissue_requested"):
        notes.append(f"No GTEx tissue matched '{evidence['tissue_requested']}'; expression was not assessed.")
    return "\n".join(lines + notes)
