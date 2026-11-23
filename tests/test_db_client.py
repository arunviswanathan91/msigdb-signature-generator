import pytest
import requests

from db_client import (
    BackendUnavailable, DatabaseClient, format_evidence_table, match_tissue,
    strength_to_min_prob,
)

# A tiny undirected network: A-B-C form a module, D is linked only to X (outside).
NETWORK = {
    "A": [("B", 0.06), ("C", 0.04), ("N1", 0.03), ("N2", 0.012)],
    "B": [("A", 0.05), ("N1", 0.045), ("GAPDH", 0.04)],
    "C": [("N1", 0.02), ("GAPDH", 0.035), ("N3", 0.03)],
    "D": [("X", 0.06)],
}


class FakeDB(DatabaseClient):
    def __init__(self):
        super().__init__("https://example.invalid")
        self.calls = []

    def _json(self, method, endpoint, **kwargs):
        self.calls.append((method, endpoint))
        if endpoint.startswith("/api/gene-neighbors/"):
            gene = endpoint.rsplit("/", 1)[1]
            return [{"target": t, "probability": p, "rank": i + 1}
                    for i, (t, p) in enumerate(NETWORK.get(gene, []))]
        if endpoint == "/api/evidence/batch":
            return {"results": {g: {"pathway_count": 100, "evidence_strength": "strong"}
                                for g in kwargs["json"] if g != "D"}}
        if endpoint == "/api/expression/batch":
            return {"results": {g: {"Blood/PBMC": 0.2 if g == "C" else 25.0, "Liver": 3.0}
                                for g in kwargs["json"] if g != "D"}}
        raise AssertionError(f"unexpected endpoint {endpoint}")


def test_strength_is_monotonic_and_on_the_scale_of_the_data():
    values = [strength_to_min_prob(s) for s in (0.0, 0.25, 0.5, 0.75, 1.0)]
    assert values == sorted(values, reverse=True)          # stronger -> lower cut-off -> more genes
    assert values[0] == 0.05 and values[-1] == 0.01
    assert strength_to_min_prob(7) == 0.01 and strength_to_min_prob(-1) == 0.05


def test_network_expansion_requires_convergent_support_and_ranks_by_strength():
    db = FakeDB()
    result = db.network_expansion(["A", "B", "C"], strength=1.0, max_additions=5)
    # N1 is a neighbour of all three; GAPDH of two but is housekeeping; N2/N3 of one only.
    assert [c["gene"] for c in result] == ["N1"]
    assert result[0]["supporters"] == ["A", "B", "C"]
    assert result[0]["score"] == pytest.approx(0.03 + 0.045 + 0.02)

    with_housekeeping = db.network_expansion(["A", "B", "C"], strength=1.0, drop_housekeeping=False)
    assert [c["gene"] for c in with_housekeeping] == ["N1", "GAPDH"]
    # Genes already in the signature are never proposed.
    assert all(c["gene"] not in {"A", "B", "C"} for c in with_housekeeping)


def test_more_strength_never_yields_fewer_candidates():
    db = FakeDB()
    counts = [len(db.network_expansion(["A", "B", "C"], strength=s, max_additions=10,
                                       drop_housekeeping=False)) for s in (0.0, 0.5, 1.0)]
    assert counts == sorted(counts) and counts[0] == 0 and counts[-1] == 2


def test_neighbour_lookups_are_cached():
    db = FakeDB()
    db.network_expansion(["A", "B"], strength=1.0)
    first = len(db.calls)
    db.network_expansion(["A", "B"], strength=0.2)
    assert len(db.calls) == first


def test_signature_evidence_is_relative_to_the_signature():
    db = FakeDB()
    evidence = db.signature_evidence(["A", "B", "C", "D", "GAPDH"], tissue_context="PBMC")
    rows = evidence["genes"]
    assert evidence["tissue"] == "Blood/PBMC"
    assert rows["A"]["linked_to"] == ["B", "C"] and rows["A"]["flags"] == []
    assert rows["B"]["linked_to"] == ["A", "GAPDH"]            # links are symmetrised
    assert rows["C"]["flags"] == ["low_expression"]            # 0.2 TPM, but linked
    assert rows["D"]["flags"] == ["isolated"]                  # its only partner is outside
    assert rows["D"]["pathway_count"] is None                  # not covered != weak
    assert rows["D"]["tissue_tpm"] is None
    assert "housekeeping" in rows["GAPDH"]["flags"]
    assert evidence["cohesion"] == pytest.approx(4 / 5)

    table = format_evidence_table(evidence)
    assert "TPM_Blood/PBMC" in table and "not_covered" in table
    assert "D | 0 | not_covered | n/a | isolated" in table


def test_evidence_without_tissue_skips_expression():
    db = FakeDB()
    evidence = db.signature_evidence(["A", "B"])
    assert evidence["tissue"] is None
    assert ("POST", "/api/expression/batch") not in db.calls
    assert "TPM" not in format_evidence_table(evidence).splitlines()[0]


def test_backend_failure_is_an_error_never_negative_evidence(monkeypatch):
    db = DatabaseClient("https://example.invalid", timeout=0.1)

    def boom(*args, **kwargs):
        raise requests.ConnectionError("no route")

    monkeypatch.setattr(db.session, "request", boom)
    with pytest.raises(BackendUnavailable):
        db.signature_evidence(["A", "B"])
    with pytest.raises(BackendUnavailable):
        db.network_expansion(["A", "B"])


@pytest.mark.parametrize("query, expected", [
    ("Blood/PBMC", "Blood/PBMC"), ("pbmc", "Blood/PBMC"), ("adipose tissue", "Adipose Tissue"),
    ("Kidney cortex", "Kidney"), ("hepatocytes", "Liver"), ("unspecified", None), ("zebra", None),
    ("", None),
])
def test_match_tissue(query, expected):
    available = ["Adipose Tissue", "Blood/PBMC", "Kidney", "Liver", "Brain"]
    assert match_tissue(query, available) == expected
