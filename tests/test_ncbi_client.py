import pytest
import requests

import ncbi_client
from ncbi_client import NCBIGene, NCBIUnavailable, normalise_name

# Trimmed copies of real NCBI Gene esummary records.
DOCS = {
    "6513": {"uid": "6513", "name": "SLC2A1", "description": "solute carrier family 2 member 1",
             "otheraliases": "CSE, DYT17, GLUT, GLUT-1, GLUT1, HTLVR"},
    "29126": {"uid": "29126", "name": "CD274", "description": "CD274 molecule",
              "otheraliases": "ADMIO5, B7-H, B7H1, PD-L1, PDCD1L1, PDL1, hPD-L1"},
    "5562": {"uid": "5562", "name": "PRKAA1", "description": "protein kinase AMP-activated catalytic subunit alpha 1",
             "otheraliases": "AMPK, AMPKa1"},
    "5563": {"uid": "5563", "name": "PRKAA2", "description": "protein kinase AMP-activated catalytic subunit alpha 2",
             "otheraliases": "AMPK, AMPK2, AMPKa2"},
    "7157": {"uid": "7157", "name": "TP53", "description": "tumor protein p53", "otheraliases": "BCC7, LFS1, P53, TRP53"},
    "999": {"uid": "999", "name": "P53", "description": "a made-up gene that lists nothing", "otheraliases": ""},
}


class FakeResponse:
    def __init__(self, payload, status=200):
        self._payload, self.status_code = payload, status

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


class FakeSession:
    def __init__(self, ids, failures=()):
        self.ids, self.failures, self.requests = ids, list(failures), []

    def get(self, url, params=None, timeout=None):
        self.requests.append((url.rsplit("/", 1)[1], params))
        if self.failures:
            failure = self.failures.pop(0)
            if isinstance(failure, Exception):
                raise failure
            return FakeResponse({}, failure)
        if url.endswith("esearch.fcgi"):
            return FakeResponse({"esearchresult": {"idlist": self.ids}})
        wanted = params["id"].split(",")
        return FakeResponse({"result": {"uids": wanted, **{u: DOCS[u] for u in wanted}}})


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(ncbi_client.time, "sleep", lambda seconds: None)


def test_names_are_matched_ignoring_case_and_punctuation():
    assert normalise_name("PD-L1") == normalise_name("pdl1") == normalise_name("PD L1") == "PDL1"
    assert normalise_name("4-1BB") == "41BB"


def test_lookup_resolves_aliases_with_two_requests():
    session = FakeSession(["6513", "29126", "5562", "5563"])
    table = NCBIGene(session=session, api_key="").lookup(["GLUT1", "PD-L1", "pdl1", "AMPK", "NOPE1"])
    assert [h["symbol"] for h in table["GLUT1"]] == ["SLC2A1"]
    assert table["GLUT1"][0]["match"] == "alias" and table["GLUT1"][0]["gene_id"] == "6513"
    assert [h["symbol"] for h in table["PD-L1"]] == [h["symbol"] for h in table["pdl1"]] == ["CD274"]
    assert [h["symbol"] for h in table["AMPK"]] == ["PRKAA1", "PRKAA2"]      # ambiguous: both returned
    assert table["NOPE1"] == []
    assert [name for name, _ in session.requests] == ["esearch.fcgi", "esummary.fcgi"]

    search = session.requests[0][1]
    assert '"GLUT1"[gene]' in search["term"] and '"Homo sapiens"[Organism]' in search["term"]
    assert search["tool"] == "msigdb-signature-generator" and search["retmode"] == "json"
    assert "email" not in search and "api_key" not in search                # nothing personal is sent


def test_an_official_symbol_beats_other_genes_that_list_it_as_an_alias():
    table = NCBIGene(session=FakeSession(["7157", "999"]), api_key="").lookup(["p53"])
    assert [(h["symbol"], h["match"]) for h in table["p53"]] == [("P53", "official")]


def test_unsafe_names_are_never_sent():
    session = FakeSession([])
    table = NCBIGene(session=session, api_key="").lookup(['BAD"[gene] OR 1=1', "ok[sym]", ""])
    assert all(hits == [] for hits in table.values())
    assert session.requests == []                                           # no query was built at all


def test_api_key_is_used_when_configured_and_raises_the_request_rate(monkeypatch):
    session = FakeSession(["6513"])
    client = NCBIGene(session=session, api_key="secret")
    client.lookup(["GLUT1"])
    assert session.requests[0][1]["api_key"] == "secret"
    assert client.min_interval < NCBIGene(session=session, api_key="").min_interval
    monkeypatch.setenv("NCBI_API_KEY", "from-env")
    assert NCBIGene(session=session).api_key == "from-env"


def test_transient_failures_are_retried_and_persistent_ones_reported():
    session = FakeSession(["6513"], failures=[429, requests.ConnectionError("reset")])
    table = NCBIGene(session=session, api_key="").lookup(["GLUT1"])
    assert [h["symbol"] for h in table["GLUT1"]] == ["SLC2A1"]

    with pytest.raises(NCBIUnavailable, match="HTTP 503"):
        NCBIGene(session=FakeSession([], failures=[503, 503, 503]), api_key="").lookup(["GLUT1"])
    with pytest.raises(NCBIUnavailable, match="HTTP 400"):
        NCBIGene(session=FakeSession([], failures=[400]), api_key="").lookup(["GLUT1"])
