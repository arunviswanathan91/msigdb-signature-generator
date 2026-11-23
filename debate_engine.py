"""
Evidence-grounded structured debate for gene-signature validation
=================================================================

Three LLM agents with opposed burdens of proof vote on EVERY gene of a
signature. Unlike the free-text debate this replaces, the unit of analysis is
the (agent, gene) vote, so every statistic below is defined on votes and not
on prose.

PROTOCOL
--------
Round 1   Each agent independently receives the research question, the
          mechanism, the full gene list and a per-gene database evidence
          table, and returns keep/remove + confidence for every gene.
Round r   Only *contested* genes (agents not unanimous) are reopened. Each
          agent sees every agent's previous vote and reason for those genes
          and votes again. Settled genes are carried forward unchanged.
Stop      when no gene is contested, agreement reaches the threshold, a round
          changes no vote (stalemate), or ``max_rounds`` is reached.

AGGREGATION (per gene g, agents a that voted)
---------------------------------------------
    P_a(g)   = c        if a votes remove          (c = stated confidence,
             = 1 - c    if a votes keep             clipped to [0.5, 1])
    p(g)     = mean_a P_a(g)                        linear opinion pool
    remove g iff p(g) >= threshold (default 0.6: clearly above a coin flip)
                 AND a strict majority voted remove
                 AND at least two agents voted on g (quorum)

METRICS
-------
    entropy(g)   = H2(p(g))                 binary Shannon entropy, bits
    conflict(g)  = 4 * Var_a P_a(g)         0 = identical, ->1 = opposed
    stability    = mean_g |1 - 2 p(g)| * (1 - conflict(g))
    agreement    = fraction of genes with unanimous final votes
    fleiss_kappa = chance-corrected agreement (None when undefined)
    vote_flips   = votes changed between consecutive rounds
    database concordance: removals with / without a database flag, and
                 flagged genes the agents kept - a direct measure of how far
                 the consensus departs from the database.

Agents equal-weighted; temperature 0 and a fixed seed by default.
"""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from db_client import format_evidence_table
from llm_client import achat_json
from model_registry import bare_model, debate_diversity

DEBATE_VERSION = "3.0.0"

# A gene is voted out only when the pooled probability of removal is clearly
# above chance. At 0.5, a 2-1 split with lukewarm confidences (pooled 0.50-0.55)
# removed genes the agents had not really agreed on.
DEFAULT_REMOVAL_THRESHOLD = 0.6

SHARED_RULES = (
    "You are one of three independent reviewers validating a gene signature: a gene set "
    "whose combined expression is meant to mark the activity of ONE biological mechanism.\n"
    "MEMBERSHIP RULE. A gene BELONGS if, in the stated context, it is a component of the "
    "mechanism, or a direct regulator, ligand, receptor, effector or transcriptional target "
    "of it, or an established marker of it. A gene DOES NOT BELONG if its only link is "
    "generic (a housekeeping gene, a broad signalling hub, a gene active in most immune or "
    "metabolic processes), or if it acts in a different process or cell type.\n"
    "Judge the mechanism as described, not only its title: direct partners of the named "
    "molecules count. Use the database evidence table together with established biology; "
    "the table is evidence, not ground truth. Never invent genes: vote only on the gene "
    "symbols listed. Respond with JSON only."
)

ROLE_PROMPTS = {
    "skeptic": (
        "ROLE - SKEPTIC. The burden of proof is on INCLUSION. Vote 'remove' unless the "
        "database evidence or well-established biology shows the gene meets the membership "
        "rule. Be alert to housekeeping genes, generic hubs and genes isolated from the "
        "rest of the signature."
    ),
    "discoverer": (
        "ROLE - ADVOCATE. The burden of proof is on EXCLUSION. Vote 'remove' only when "
        "the gene clearly fails the membership rule. A gene with a plausible, documented "
        "role should be kept even if the database evidence is thin - databases lag the "
        "literature."
    ),
    "mediator": (
        "ROLE - MEDIATOR. Apply no prior in either direction. Weigh database evidence "
        "and biological plausibility equally and decide each gene on balance."
    ),
}

ROLE_LABELS = {"skeptic": "Skeptic", "discoverer": "Advocate", "mediator": "Mediator"}

DB_FLAGS = ("isolated", "low_expression", "housekeeping")


# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class GeneVote:
    gene: str
    vote: str            # 'keep' | 'remove'
    confidence: float    # confidence in the vote, [0.5, 1]
    reason: str = ""

    @property
    def p_remove(self) -> float:
        return self.confidence if self.vote == "remove" else 1.0 - self.confidence


@dataclass
class AgentTurn:
    role: str
    model: str
    round_num: int
    votes: Dict[str, GeneVote] = field(default_factory=dict)   # cumulative position
    revoted: List[str] = field(default_factory=list)           # genes voted on this round
    out_of_set: List[str] = field(default_factory=list)        # symbols not in the signature
    missing: List[str] = field(default_factory=list)           # asked but not answered
    error: Optional[str] = None
    llm: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            'role': self.role, 'model': self.model, 'round': self.round_num,
            'error': self.error,
            'votes': {g: {'vote': v.vote, 'confidence': round(v.confidence, 3), 'reason': v.reason}
                      for g, v in self.votes.items() if g in self.revoted},
            'out_of_set_mentions': self.out_of_set,
            'unanswered': self.missing,
            'llm': self.llm,
        }


@dataclass
class DebateRound:
    round_num: int
    turns: List[AgentTurn]
    agreement: float
    contested: List[str]
    flips: int = 0


@dataclass
class GeneDecision:
    gene: str
    p_remove: Optional[float]
    decision: str                 # 'keep' | 'remove' | 'no_quorum'
    entropy: Optional[float]
    conflict: Optional[float]
    votes: Dict[str, Dict]        # role -> {vote, confidence, reason}
    n_remove: int = 0
    n_voters: int = 0
    db_flags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        def rounded(x):
            return None if x is None else round(x, 4)
        return {
            'gene': self.gene, 'decision': self.decision,
            'p_remove': rounded(self.p_remove), 'entropy': rounded(self.entropy),
            'conflict': rounded(self.conflict), 'votes_remove': self.n_remove,
            'voters': self.n_voters, 'db_flags': self.db_flags, 'votes': self.votes,
        }


@dataclass
class DebateResult:
    genes: List[str]
    models: Dict[str, str]
    rounds: List[DebateRound]
    decisions: Dict[str, GeneDecision]
    metrics: Dict
    parameters: Dict
    evidence: Optional[Dict] = None

    @property
    def genes_to_remove(self) -> List[str]:
        return [g for g in self.genes if self.decisions[g].decision == "remove"]

    @property
    def total_rounds(self) -> int:
        return len(self.rounds)

    @property
    def failed(self) -> bool:
        """True if fewer than two agents ever produced votes."""
        return self.metrics.get('agents_responding', 0) < 2

    def to_dict(self) -> Dict:
        return {
            'debate_version': DEBATE_VERSION,
            'models': self.models,
            'parameters': self.parameters,
            'metrics': self.metrics,
            'genes_removed': self.genes_to_remove,
            'gene_decisions': [self.decisions[g].to_dict() for g in self.genes],
            'rounds': [
                {'round': r.round_num, 'agreement': round(r.agreement, 4),
                 'contested': r.contested, 'vote_flips': r.flips,
                 'turns': [t.to_dict() for t in r.turns]}
                for r in self.rounds
            ],
            'database_evidence': self.evidence,
        }


# ============================================================
# PURE STATISTICS (unit-tested, no LLM)
# ============================================================

def binary_entropy(p: float) -> float:
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))


def clip_confidence(value) -> float:
    """Coerce a model-reported confidence to [0.5, 1]. Accepts 0-1 or 0-100."""
    try:
        conf = float(value)
    except (TypeError, ValueError):
        return 0.5
    if math.isnan(conf):
        return 0.5
    if conf > 1.0:
        conf = conf / 100.0 if conf <= 100.0 else 1.0
    return min(1.0, max(0.5, conf))


def parse_votes(payload, genes: Sequence[str]) -> tuple:
    """
    Read an agent's JSON into votes for genes in ``genes``.

    Accepts the compact opening shape {"keep": {GENE: conf}, "remove": {GENE:
    {confidence, reason}}}, the rebuttal shape {"votes": [{gene, vote,
    confidence, reason}]}, {"votes": {GENE: {...}}} or a bare list. Symbols are matched
    case-insensitively; anything not in the signature is returned as
    out-of-set and never counted.

    Returns (votes: {gene: GeneVote}, out_of_set: [symbol]).
    """
    canonical = {g.upper(): g for g in genes}
    votes: Dict[str, GeneVote] = {}
    out_of_set: List[str] = []

    if isinstance(payload, dict) and "votes" not in payload and ("keep" in payload or "remove" in payload):
        raw = []
        for decision in ("keep", "remove"):
            block = payload.get(decision) or {}
            if isinstance(block, list):       # ["GENE", ...] or [{"gene": ...}, ...]
                block = {(b.get("gene") if isinstance(b, dict) else b): b for b in block}
            if not isinstance(block, dict):
                continue
            for symbol, value in block.items():
                entry = dict(value) if isinstance(value, dict) else {"confidence": value}
                if not isinstance(value, (dict, int, float)):
                    entry = {}
                entry.update(gene=symbol, vote=decision)
                raw.append(entry)
    else:
        raw = payload.get("votes", payload) if isinstance(payload, dict) else payload
    if isinstance(raw, dict):
        raw = [dict(v, gene=k) if isinstance(v, dict) else {"gene": k, "vote": v}
               for k, v in raw.items()]
    if not isinstance(raw, list):
        return votes, out_of_set

    for item in raw:
        if not isinstance(item, dict):
            continue
        symbol = str(item.get("gene", "")).strip()
        decision = str(item.get("vote", "")).strip().lower()
        if not symbol or decision not in ("keep", "remove"):
            continue
        gene = canonical.get(symbol.upper())
        if gene is None:
            out_of_set.append(symbol)
            continue
        votes[gene] = GeneVote(
            gene=gene, vote=decision,
            confidence=clip_confidence(item.get("confidence", 0.5)),
            reason=str(item.get("reason") or "").strip()[:300],
        )
    return votes, out_of_set


def aggregate(genes: Sequence[str], positions: Dict[str, Dict[str, GeneVote]],
              threshold: float = DEFAULT_REMOVAL_THRESHOLD, evidence: Optional[Dict] = None
              ) -> Dict[str, GeneDecision]:
    """Pool the agents' final positions into one decision per gene."""
    decisions: Dict[str, GeneDecision] = {}
    for gene in genes:
        cast = {role: votes[gene] for role, votes in positions.items() if gene in votes}
        flags = []
        if evidence:
            flags = list(evidence.get("genes", {}).get(gene, {}).get("flags", []))
        record = {role: {'vote': v.vote, 'confidence': round(v.confidence, 3), 'reason': v.reason}
                  for role, v in cast.items()}

        if len(cast) < 2:
            decisions[gene] = GeneDecision(gene, None, "no_quorum", None, None, record,
                                           n_voters=len(cast), db_flags=flags)
            continue

        probs = [v.p_remove for v in cast.values()]
        p = sum(probs) / len(probs)
        variance = sum((x - p) ** 2 for x in probs) / len(probs)
        n_remove = sum(1 for v in cast.values() if v.vote == "remove")
        remove = p >= threshold and n_remove * 2 > len(cast)

        decisions[gene] = GeneDecision(
            gene=gene, p_remove=p, decision="remove" if remove else "keep",
            entropy=binary_entropy(p), conflict=min(1.0, 4.0 * variance),
            votes=record, n_remove=n_remove, n_voters=len(cast), db_flags=flags,
        )
    return decisions


def contested_genes(genes: Sequence[str], positions: Dict[str, Dict[str, GeneVote]]) -> List[str]:
    """Genes on which the agents that voted are not unanimous."""
    out = []
    for gene in genes:
        cast = {votes[gene].vote for votes in positions.values() if gene in votes}
        if len(cast) > 1:
            out.append(gene)
    return out


def agreement_rate(genes: Sequence[str], positions: Dict[str, Dict[str, GeneVote]]) -> float:
    """Fraction of genes (with >= 2 votes) on which all voting agents agree."""
    judged = unanimous = 0
    for gene in genes:
        cast = [votes[gene].vote for votes in positions.values() if gene in votes]
        if len(cast) >= 2:
            judged += 1
            unanimous += len(set(cast)) == 1
    return unanimous / judged if judged else 0.0


def fleiss_kappa(genes: Sequence[str], positions: Dict[str, Dict[str, GeneVote]]) -> Optional[float]:
    """
    Fleiss' kappa over genes rated by all agents. None when it is undefined
    (fewer than two raters, no fully rated gene, or every vote identical).
    """
    raters = len(positions)
    if raters < 2:
        return None
    rows = []
    for gene in genes:
        cast = [votes[gene].vote for votes in positions.values() if gene in votes]
        if len(cast) == raters:
            rows.append(sum(1 for v in cast if v == "remove"))
    if not rows:
        return None
    n = len(rows)
    p_remove = sum(rows) / (n * raters)
    expected = p_remove ** 2 + (1 - p_remove) ** 2
    if expected >= 1.0:
        return None
    observed = sum(
        (k * (k - 1) + (raters - k) * (raters - k - 1)) / (raters * (raters - 1)) for k in rows
    ) / n
    return (observed - expected) / (1 - expected)


def summarise(genes: Sequence[str], decisions: Dict[str, GeneDecision],
              positions: Dict[str, Dict[str, GeneVote]], rounds: Sequence[DebateRound],
              stop_reason: str) -> Dict:
    """Signature-level metrics from per-gene decisions."""
    judged = [d for d in decisions.values() if d.p_remove is not None]
    removed = [d for d in judged if d.decision == "remove"]
    kappa = fleiss_kappa(genes, positions)

    def mean(values):
        values = list(values)
        return round(sum(values) / len(values), 4) if values else None

    flagged = [d for d in judged if d.db_flags]
    return {
        'agents_responding': len(positions),
        'genes_judged': len(judged),
        'genes_without_quorum': len(decisions) - len(judged),
        'genes_removed': len(removed),
        'stability': mean(abs(1 - 2 * d.p_remove) * (1 - d.conflict) for d in judged),
        'entropy': mean(d.entropy for d in judged),
        'conflict': mean(d.conflict for d in judged),
        'agreement': round(agreement_rate(genes, positions), 4),
        'fleiss_kappa': None if kappa is None else round(kappa, 4),
        'rounds': len(rounds),
        'stop_reason': stop_reason,
        'vote_flips': sum(r.flips for r in rounds),
        'out_of_set_mentions': sum(len(t.out_of_set) for r in rounds for t in r.turns),
        'database_concordance': {
            'flagged_genes': len(flagged),
            'flagged_and_removed': sum(1 for d in flagged if d.decision == "remove"),
            'flagged_but_kept': sum(1 for d in flagged if d.decision == "keep"),
            'removed_without_flag': sum(1 for d in removed if not d.db_flags),
        },
    }


# ============================================================
# PROMPTS
# ============================================================

# Opening round: compact, so a 20-gene signature fits in a few hundred tokens.
OPENING_SCHEMA = (
    '{"keep": {"SYMBOL": confidence, ...}, '
    '"remove": {"SYMBOL": {"confidence": 0.5-1.0, "reason": "max 10 words"}, ...}}'
)
# Rebuttal rounds: few genes, and the argument for KEEPING matters too.
VOTE_SCHEMA = (
    '{"votes": [{"gene": "SYMBOL", "vote": "keep" | "remove", '
    '"confidence": 0.5-1.0, "reason": "max 20 words"}]}'
)


def build_opening_prompt(genes: Sequence[str], *, query: str, mechanism: str,
                         description: str, context: Optional[Dict],
                         evidence: Optional[Dict]) -> str:
    parts = [f"RESEARCH QUESTION: {query or 'not provided'}",
             f"MECHANISM UNDER REVIEW: {mechanism}"]
    if description:
        parts.append(f"MECHANISM DESCRIPTION: {description}")
    if context:
        bits = [f"{k}: {context[k]}" for k in ('species', 'tissue', 'disease', 'cell_type', 'treatment')
                if context.get(k)]
        if bits:
            parts.append("BIOLOGICAL CONTEXT: " + "; ".join(bits))
    parts.append(f"\nSIGNATURE ({len(genes)} genes): {', '.join(genes)}")
    if evidence:
        parts.append("\nDATABASE EVIDENCE (one row per gene):\n" + format_evidence_table(evidence))
    else:
        parts.append("\nDATABASE EVIDENCE: unavailable for this run. Rely on established biology "
                     "and lower your confidence accordingly.")
    parts.append(
        f"\nTASK: Vote on ALL {len(genes)} genes. 'remove' means the gene fails the membership "
        "rule for this mechanism in this context. 'confidence' is how sure you are that your vote is "
        "right (0.5 = coin flip, 1.0 = certain). Every gene must appear exactly once, under "
        "either 'keep' or 'remove'.\n"
        f"Return exactly this JSON shape, minified on one line:\n{OPENING_SCHEMA}"
    )
    return "\n".join(parts)


def build_recap(genes: Sequence[str], *, query: str, mechanism: str, description: str,
                context: Optional[Dict]) -> str:
    """What is being judged, without the evidence table - context for rebuttals."""
    parts = [f"RESEARCH QUESTION: {query or 'not provided'}",
             f"MECHANISM UNDER REVIEW: {mechanism}"]
    if description:
        parts.append(f"MECHANISM DESCRIPTION: {description}")
    if context:
        bits = [f"{k}: {context[k]}" for k in ('species', 'tissue', 'disease', 'cell_type', 'treatment')
                if context.get(k)]
        if bits:
            parts.append("BIOLOGICAL CONTEXT: " + "; ".join(bits))
    parts.append(f"SIGNATURE ({len(genes)} genes): {', '.join(genes)}")
    return "\n".join(parts)


def build_rebuttal_prompt(contested: Sequence[str], positions: Dict[str, Dict[str, GeneVote]],
                          role: str, round_num: int, evidence: Optional[Dict]) -> str:
    lines = [
        f"ROUND {round_num}. The reviewers disagree on {len(contested)} gene(s). All other "
        "genes are settled. Positions from the previous round:", ""]
    for gene in contested:
        lines.append(f"{gene}")
        if evidence and gene in evidence.get("genes", {}):
            row = evidence["genes"][gene]
            lines.append(f"  database: links_to_signature={row['links_in_signature']}, "
                         f"flags={','.join(row['flags']) or 'none'}")
        for other, votes in positions.items():
            if gene not in votes:
                continue
            vote = votes[gene]
            who = "YOU" if other == role else ROLE_LABELS.get(other, other)
            reason = f" - {vote.reason}" if vote.reason else ""
            lines.append(f"  {who}: {vote.vote} (confidence {vote.confidence:.2f}){reason}")
        lines.append("")
    lines.append(
        "TASK: Reconsider ONLY the genes listed above. Change your vote if another reviewer's "
        "argument is stronger than yours; hold it if not. Do not change a vote merely to agree.\n"
        f"Return exactly this JSON shape, minified, one entry per listed gene:\n{VOTE_SCHEMA}"
    )
    return "\n".join(lines)


# ============================================================
# ENGINE
# ============================================================

def opening_budget(n_genes: int) -> int:
    """
    Output-token budget for a round-1 vote on ``n_genes`` genes.

    Deliberately tight: providers reserve the *requested* budget against a
    per-minute allowance, so an over-generous request throttles the debate even
    when the answer is short. A truncated answer is retried with double the
    budget by ``achat_json``, so tight is safe.
    """
    return 220 + 22 * n_genes


def rebuttal_budget(n_genes: int) -> int:
    """Budget for a rebuttal round: fewer genes, but a reason for every vote."""
    return 180 + 55 * n_genes


def estimate_tokens_per_model(n_genes: int, max_rounds: int, grounded: bool = True) -> int:
    """
    Rough upper estimate of tokens ONE model spends debating one signature
    (prompt + completion, all rounds). Used to warn before a run that would
    not fit a free-tier quota; real usage is usually lower because most
    debates stop early.
    """
    opening_prompt = 420 + 6 * n_genes + (14 * n_genes + 130 if grounded else 0)
    contested = max(2, n_genes // 4)
    rebuttal_prompt = 330 + 6 * n_genes + 75 * contested
    return (opening_prompt + opening_budget(n_genes)
            + max(0, max_rounds - 1) * (rebuttal_prompt + rebuttal_budget(contested)))


class DebateEngine:
    """Runs the structured debate for one signature."""

    def __init__(
        self,
        llm,
        models: Dict[str, str],
        *,
        fallbacks: Sequence[str] = (),
        removal_threshold: float = DEFAULT_REMOVAL_THRESHOLD,
        temperature: float = 0.0,
        seed: Optional[int] = 0,
    ):
        """
        ``llm`` is an ``llm_client.LLMRouter``; ``models`` maps each role to a
        model reference, which may point at different providers - the more
        unrelated the three models, the more a consensus means.
        """
        missing = [r for r in ROLE_PROMPTS if not models.get(r)]
        if missing:
            raise ValueError(f"No model resolved for debate role(s): {', '.join(missing)}")
        self.llm = llm
        self.models = {role: models[role] for role in ROLE_PROMPTS}
        self.fallbacks = list(fallbacks)
        self.removal_threshold = removal_threshold
        self.temperature = temperature
        self.seed = seed

    async def _ask(self, pool, role: str, messages: List[Dict], genes: Sequence[str],
                   round_num: int, carried: Dict[str, GeneVote]) -> AgentTurn:
        turn = AgentTurn(role=role, model=self.models[role], round_num=round_num,
                         votes=dict(carried))
        try:
            # If this role's model is retired or out of quota, prefer a model no
            # other role uses; only then share one. The model actually used is
            # recorded on the turn, so reduced independence is visible.
            in_use = {bare_model(m) for m in self.models.values()}
            spare = [m for m in self.fallbacks if bare_model(m) not in in_use]
            shared = [m for m in self.models.values() if m != self.models[role]]
            result = await achat_json(
                pool, self.models[role], messages,
                max_tokens=opening_budget(len(genes)) if round_num == 1 else rebuttal_budget(len(genes)),
                temperature=self.temperature, seed=self.seed,
                fallbacks=list(dict.fromkeys(spare + shared)),
            )
        except Exception as exc:  # noqa: BLE001 - one agent failing must not end the debate
            turn.error = str(exc)[:300]
            return turn

        votes, out_of_set = parse_votes(result.data, genes)
        turn.votes.update(votes)
        turn.revoted = list(votes)
        turn.out_of_set = out_of_set
        turn.missing = [g for g in genes if g not in votes]
        turn.model = result.model
        turn.llm = result.provenance()
        if not votes:
            turn.error = "response contained no usable votes"
        return turn

    async def arun(
        self,
        genes: Sequence[str],
        evidence: Optional[Dict] = None,
        *,
        query: str = "",
        mechanism: str = "",
        description: str = "",
        context: Optional[Dict] = None,
        max_rounds: int = 3,
        agreement_threshold: float = 1.0,
    ) -> DebateResult:
        genes = list(dict.fromkeys(genes))
        opening = build_opening_prompt(genes, query=query, mechanism=mechanism,
                                       description=description, context=context,
                                       evidence=evidence)
        recap = build_recap(genes, query=query, mechanism=mechanism,
                            description=description, context=context)

        positions: Dict[str, Dict[str, GeneVote]] = {}
        rounds: List[DebateRound] = []
        stop_reason = "max_rounds"

        async with self.llm.async_pool() as pool:
            for round_num in range(1, max(1, max_rounds) + 1):
                if round_num == 1:
                    targets = genes
                    jobs = [
                        self._ask(pool, role, [
                            {"role": "system", "content": f"{SHARED_RULES}\n\n{ROLE_PROMPTS[role]}"},
                            {"role": "user", "content": opening},
                        ], targets, round_num, {})
                        for role in self.models
                    ]
                else:
                    targets = contested_genes(genes, positions)
                    # The rebuttal carries the database rows of the contested
                    # genes itself, so the full evidence table is not resent.
                    jobs = [
                        self._ask(pool, role, [
                            {"role": "system", "content": f"{SHARED_RULES}\n\n{ROLE_PROMPTS[role]}"},
                            {"role": "user", "content": recap},
                            {"role": "user", "content": build_rebuttal_prompt(
                                targets, positions, role, round_num, evidence)},
                        ], targets, round_num, positions.get(role, {}))
                        for role in self.models if role in positions
                    ]

                turns = await asyncio.gather(*jobs)

                flips = 0
                for turn in turns:
                    if not turn.votes:
                        continue
                    previous = positions.get(turn.role, {})
                    flips += sum(1 for g in turn.revoted
                                 if g in previous and previous[g].vote != turn.votes[g].vote)
                    positions[turn.role] = turn.votes

                still_contested = contested_genes(genes, positions)
                agreement = agreement_rate(genes, positions)
                rounds.append(DebateRound(round_num, list(turns), agreement, still_contested, flips))

                if len(positions) < 2:
                    stop_reason = "insufficient_agents"
                    break
                if not still_contested:
                    stop_reason = "unanimous"
                    break
                if agreement >= agreement_threshold:
                    stop_reason = "agreement_threshold"
                    break
                if round_num > 1 and flips == 0:
                    stop_reason = "stalemate"
                    break

        decisions = aggregate(genes, positions, self.removal_threshold, evidence)
        metrics = summarise(genes, decisions, positions, rounds, stop_reason)
        return DebateResult(
            genes=genes,
            models={role: next((t.model for r in rounds for t in r.turns if t.role == role),
                               self.models[role]) for role in self.models},
            rounds=rounds, decisions=decisions, metrics=metrics, evidence=evidence,
            parameters={
                'max_rounds': max_rounds, 'agreement_threshold': agreement_threshold,
                'removal_threshold': self.removal_threshold,
                'temperature': self.temperature, 'seed': self.seed,
                'database_grounded': evidence is not None,
                'configured_models': self.models,
                'diversity': debate_diversity(self.models),
            },
        )

    def run(self, genes: Sequence[str], evidence: Optional[Dict] = None, **kwargs) -> DebateResult:
        """Synchronous wrapper. Uses its own event loop, so it is safe to call
        from Streamlit's script thread."""
        return asyncio.run(self.arun(genes, evidence, **kwargs))
