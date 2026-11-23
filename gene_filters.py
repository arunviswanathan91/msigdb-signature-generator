"""
Housekeeping-gene filter - the single definition used by every layer.

Previously three different lists lived in three files, and ribosomal genes
were matched with a bare ``startswith('RPS')`` that also caught kinases such
as RPS6KA5 (MSK1) and RPS6KB1 (S6K1).
"""

import re
from typing import Iterable, List, Tuple

HOUSEKEEPING_GENES = frozenset({
    # Glycolysis / core metabolism
    'GAPDH', 'GAPDHS', 'PGK1', 'ENO1', 'PKM', 'ALDOA', 'TPI1',
    'LDHA', 'LDHB', 'G6PD', 'PFKM', 'PFKL',
    # Cytoskeleton
    'ACTB', 'ACTG1', 'TUBB', 'TUBA1A', 'TUBA1B', 'TUBA1C',
    'TUBB4B', 'TUBA4A', 'TUBB2A', 'TUBB3',
    # Classic reference genes / translation
    'B2M', 'PPIA', 'PPIB', 'HPRT1', 'TBP', 'YWHAZ', 'UBC', 'EEF1A1',
    'RPLP0', 'RPLP1', 'RPLP2', 'RPSA',
    # Chaperones
    'HSP90AB1', 'HSPA8', 'HSPD1', 'HSPA1A', 'HSP90AA1',
})

# Ribosomal proteins with documented disease-specific roles are kept.
RIBOSOMAL_EXCEPTIONS = frozenset({'RPL22', 'RPS19', 'RPS24'})

# RPL13A, RPS27A, RPS4X, RPS4Y1, RPL22L1 match; RPS6KA5 / RPS6KB1 do not.
_RIBOSOMAL = re.compile(r'^(?:RP[LS]\d+[A-Z]?\d*|MRP[LS]\d+)$')
_MITOCHONDRIAL = re.compile(r'^MT-')


def is_housekeeping(gene: str) -> bool:
    """True for constitutively expressed genes that carry no mechanism signal."""
    if gene in HOUSEKEEPING_GENES:
        return True
    if gene in RIBOSOMAL_EXCEPTIONS:
        return False
    return bool(_RIBOSOMAL.match(gene) or _MITOCHONDRIAL.match(gene))


def split_housekeeping(genes: Iterable[str]) -> Tuple[List[str], List[str]]:
    """Return (non_housekeeping, housekeeping), both in input order."""
    kept, flagged = [], []
    for gene in genes:
        (flagged if is_housekeeping(gene) else kept).append(gene)
    return kept, flagged
