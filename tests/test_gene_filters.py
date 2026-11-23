import pytest

from gene_filters import is_housekeeping, split_housekeeping


@pytest.mark.parametrize("gene", [
    "GAPDH", "ACTB", "B2M", "RPLP0", "RPSA",
    "RPL13A", "RPS27A", "RPS4X", "RPS4Y1", "RPL22L1", "RPS2", "RPL5",
    "MRPL1", "MRPS35", "MT-CO1", "MT-ND4",
])
def test_housekeeping_genes_are_flagged(gene):
    assert is_housekeeping(gene)


@pytest.mark.parametrize("gene", [
    # Kinases that merely START with RPS - the old prefix match caught these.
    "RPS6KA5", "RPS6KB1", "RPS6KA1",
    # Documented disease-relevant ribosomal exceptions.
    "RPL22", "RPS19", "RPS24",
    # Ordinary genes.
    "PDCD1", "TP53", "MTOR", "MT1A", "RPE65", "RPTOR",
])
def test_non_housekeeping_genes_are_kept(gene):
    assert not is_housekeeping(gene)


def test_split_preserves_order():
    kept, flagged = split_housekeeping(["PDCD1", "GAPDH", "RPS6KA5", "RPL5", "TOX"])
    assert kept == ["PDCD1", "RPS6KA5", "TOX"]
    assert flagged == ["GAPDH", "RPL5"]
