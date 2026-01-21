"""
Multi-Round Debate System with Database Injector (GROQ EDITION)
================================================================

Version: 2.0.0 - Migrated to Groq API

Orchestrates debates between 3 LLMs with database grounding for:
1. Signature Validation: Debate whether to REMOVE genes
2. Gene Expansion: Debate whether to ADD candidate genes

Features:
- Configurable rounds (1-20)
- Database injector between rounds
- Weighted voting by confidence
- Convergence tracking
- Conversational output formatting
- Groq API for ultra-fast inference
"""

import asyncio
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import re
import time
from openai import AsyncOpenAI  # CHANGED: Use OpenAI Client for Groq


class DebateMode(Enum):
    """Debate mode selection"""
    VALIDATION = "validation"  # Debate entire signature, remove bad genes
    EXPANSION = "expansion"    # Debate single candidate, add or reject


@dataclass
class DebateMessage:
    """Single message in debate conversation"""
    round_num: int
    speaker: str  # "qwen", "zephyr", "phi", "injector", "consensus"
    message: str
    db_sources: List[str] = field(default_factory=list)
    confidence: Optional[float] = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class DebateRound:
    """Complete round of debate"""
    round_num: int
    messages: List[DebateMessage]
    convergence_rate: float = 0.0
    genes_to_remove: Set[str] = field(default_factory=set)
    genes_to_add: Set[str] = field(default_factory=set)


@dataclass
class DebateResult:
    """Final debate outcome"""
    mode: DebateMode
    total_rounds: int
    final_decision: str  # "remove", "add", "keep", "reject"
    affected_genes: List[str]
    confidence: float
    convergence_rate: float
    all_rounds: List[DebateRound]


class DatabaseInjector:
    """
    Injects database-grounded context into debates.
    
    Acts as an impartial fact-provider between LLM rounds.
    """
    
    def __init__(self, db_client):
        """
        Args:
            db_client: DatabaseClientEnhanced instance
        """
        self.db_client = db_client
    
    def inject_validation_context(
        self,
        genes: List[str],
        tissue_context: Optional[str] = None
    ) -> DebateMessage:
        """
        Inject context for signature validation debate.
        
        Probes entire signature and returns aggregate issues.
        """
        issues = self.db_client.probe_signature_for_validation(
            genes=genes,
            tissue_context=tissue_context
        )
        
        from db_client_enhanced import format_validation_issues_for_debate
        formatted_text = format_validation_issues_for_debate(issues)
        
        return DebateMessage(
            round_num=0,  # Will be set by caller
            speaker="injector",
            message=formatted_text,
            db_sources=issues['sources_used']
        )
    
    def inject_expansion_context(
        self,
        candidate_gene: str,
        existing_genes: List[str]
    ) -> DebateMessage:
        """
        Inject context for gene expansion debate.
        
        Probes single candidate gene in detail.
        """
        context = self.db_client.get_expansion_candidate_context(
            candidate_gene=candidate_gene,
            existing_genes=existing_genes
        )
        
        from db_client_enhanced import format_expansion_context_for_debate
        formatted_text = format_expansion_context_for_debate(context)
        
        return DebateMessage(
            round_num=0,
            speaker="injector",
            message=formatted_text,
            db_sources=context['sources']
        )


class MultiRoundDebateEngine:
    """
    Orchestrates multi-round debates with database grounding.

    Architecture:
    1. Round 1: Injector provides initial database context
    2. LLMs debate based on context
    3. Round N: Injector provides follow-up data if needed
    4. LLMs refine positions
    5. Meta-synthesizer makes final decision
    """

    def __init__(
        self,
        api_key: str,
        db_client,
        base_url: str = "https://api.groq.com/openai/v1",
        model_configs: Optional[Dict] = None
    ):
        """
        Args:
            api_key: Groq API key
            db_client: DatabaseClientEnhanced instance
            base_url: Groq API base URL (default: https://api.groq.com/openai/v1)
            model_configs: Optional dict of model parameters
        """
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.db_client = db_client
        self.injector = DatabaseInjector(db_client)

        # GROQ MODELS MAPPING
        # Using different Groq models for diverse perspectives in debate
        self.models = {
            "qwen": "llama-3.3-70b-versatile",    # Smartest, most capable
            "zephyr": "llama-3.1-8b-instant",     # Fast and creative
            "phi": "gemma2-9b-it"                 # Alternative perspective
        }

        # Override with custom configs if provided
        if model_configs:
            self.models.update(model_configs)

        # Model reliability weights (can be tuned)
        self.reliability_weights = {
            "qwen": 1.2,
            "zephyr": 1.0,
            "phi": 0.9
        }
    
    async def run_validation_debate(
        self,
        genes: List[str],
        tissue_context: Optional[str] = None,
        max_rounds: int = 10,
        convergence_threshold: float = 0.85
    ) -> DebateResult:
        """
        Run multi-round validation debate.
        
        Goal: Decide which genes to REMOVE from signature.
        
        Args:
            genes: Current signature genes
            tissue_context: Optional tissue for validation
            max_rounds: Maximum debate rounds
            convergence_threshold: Stop when convergence ≥ this
        
        Returns:
            DebateResult with genes_to_remove
        """
        all_rounds = []
        current_genes_to_remove = set()
        
        for round_num in range(1, max_rounds + 1):
            round_messages = []
            
            # Inject database context (round 1 or when needed)
            if round_num == 1 or round_num % 3 == 0:
                injector_msg = self.injector.inject_validation_context(
                    genes=genes,
                    tissue_context=tissue_context
                )
                injector_msg.round_num = round_num
                round_messages.append(injector_msg)
            
            # Each LLM responds
            llm_messages = await self._collect_llm_responses_validation(
                round_num=round_num,
                genes=genes,
                previous_rounds=all_rounds,
                injector_context=round_messages[0] if round_messages else None
            )
            round_messages.extend(llm_messages)
            
            # Parse votes
            votes = self._parse_removal_votes(llm_messages)
            current_genes_to_remove = set(votes.keys())
            
            # Calculate convergence
            convergence = self._calculate_convergence(llm_messages)
            
            # Create round record
            debate_round = DebateRound(
                round_num=round_num,
                messages=round_messages,
                convergence_rate=convergence,
                genes_to_remove=current_genes_to_remove
            )
            all_rounds.append(debate_round)
            
            # Check convergence
            if convergence >= convergence_threshold:
                break
        
        # Meta-synthesizer makes final decision
        final_decision, confidence = self._synthesize_final_decision(
            all_rounds,
            mode=DebateMode.VALIDATION
        )
        
        return DebateResult(
            mode=DebateMode.VALIDATION,
            total_rounds=len(all_rounds),
            final_decision=final_decision,
            affected_genes=list(current_genes_to_remove),
            confidence=confidence,
            convergence_rate=all_rounds[-1].convergence_rate,
            all_rounds=all_rounds
        )
    
    async def run_expansion_debate(
        self,
        candidate_gene: str,
        existing_genes: List[str],
        max_rounds: int = 10,
        convergence_threshold: float = 0.85
    ) -> DebateResult:
        """
        Run multi-round expansion debate.
        
        Goal: Decide whether to ADD candidate gene.
        
        Args:
            candidate_gene: Gene to evaluate
            existing_genes: Current signature
            max_rounds: Maximum rounds
            convergence_threshold: Stop when converged
        
        Returns:
            DebateResult with decision (add/reject)
        """
        all_rounds = []
        
        for round_num in range(1, max_rounds + 1):
            round_messages = []
            
            # Inject context (round 1 only for expansion)
            if round_num == 1:
                injector_msg = self.injector.inject_expansion_context(
                    candidate_gene=candidate_gene,
                    existing_genes=existing_genes
                )
                injector_msg.round_num = round_num
                round_messages.append(injector_msg)
            
            # LLM responses
            llm_messages = await self._collect_llm_responses_expansion(
                round_num=round_num,
                candidate_gene=candidate_gene,
                previous_rounds=all_rounds,
                injector_context=round_messages[0] if round_messages else None
            )
            round_messages.extend(llm_messages)
            
            # Calculate convergence
            convergence = self._calculate_convergence(llm_messages)
            
            debate_round = DebateRound(
                round_num=round_num,
                messages=round_messages,
                convergence_rate=convergence
            )
            all_rounds.append(debate_round)
            
            if convergence >= convergence_threshold:
                break
        
        # Final decision
        final_decision, confidence = self._synthesize_final_decision(
            all_rounds,
            mode=DebateMode.EXPANSION
        )
        
        affected_genes = [candidate_gene] if final_decision == "add" else []
        
        return DebateResult(
            mode=DebateMode.EXPANSION,
            total_rounds=len(all_rounds),
            final_decision=final_decision,
            affected_genes=affected_genes,
            confidence=confidence,
            convergence_rate=all_rounds[-1].convergence_rate,
            all_rounds=all_rounds
        )
    
    async def _collect_llm_responses_validation(
        self,
        round_num: int,
        genes: List[str],
        previous_rounds: List[DebateRound],
        injector_context: Optional[DebateMessage]
    ) -> List[DebateMessage]:
        """
        Collect responses from all 3 LLMs for validation debate.
        """
        # Build conversation history
        history = self._build_conversation_history(previous_rounds)
        
        # Add injector context if present
        if injector_context:
            history.append({
                "role": "user",
                "content": injector_context.message
            })
        
        # Prompt for this round
        prompt = self._build_validation_prompt(genes, round_num)
        history.append({
            "role": "user",
            "content": prompt
        })
        
        # Query all LLMs in parallel
        tasks = [
            self._query_llm_async(model_name, model_id, history)
            for model_name, model_id in self.models.items()
        ]
        
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Convert to DebateMessages
        messages = []
        for model_name, response in zip(self.models.keys(), responses):
            if isinstance(response, Exception):
                print(f"⚠️ {model_name} failed: {response}")
                continue
            
            # Parse confidence from response
            confidence = self._extract_confidence(response)
            
            messages.append(DebateMessage(
                round_num=round_num,
                speaker=model_name,
                message=response,
                confidence=confidence
            ))
        
        return messages
    
    async def _collect_llm_responses_expansion(
        self,
        round_num: int,
        candidate_gene: str,
        previous_rounds: List[DebateRound],
        injector_context: Optional[DebateMessage]
    ) -> List[DebateMessage]:
        """
        Collect LLM responses for expansion debate.
        """
        history = self._build_conversation_history(previous_rounds)
        
        if injector_context:
            history.append({
                "role": "user",
                "content": injector_context.message
            })
        
        prompt = self._build_expansion_prompt(candidate_gene, round_num)
        history.append({
            "role": "user",
            "content": prompt
        })
        
        tasks = [
            self._query_llm_async(model_name, model_id, history)
            for model_name, model_id in self.models.items()
        ]
        
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        messages = []
        for model_name, response in zip(self.models.keys(), responses):
            if isinstance(response, Exception):
                continue
            
            confidence = self._extract_confidence(response)
            
            messages.append(DebateMessage(
                round_num=round_num,
                speaker=model_name,
                message=response,
                confidence=confidence
            ))
        
        return messages
    
    async def _query_llm_async(
        self,
        model_name: str,
        model_id: str,
        conversation_history: List[Dict]
    ) -> str:
        """
        Async LLM query via Groq API (OpenAI-compatible).
        """
        try:
            # Call Groq API via AsyncOpenAI client
            response = await self.client.chat.completions.create(
                model=model_id,
                messages=conversation_history,
                max_tokens=500,
                temperature=0.7
            )

            return response.choices[0].message.content
        except Exception as e:
            return f"Error ({model_name}): {str(e)}"
    
    def _build_validation_prompt(self, genes: List[str], round_num: int) -> str:
        """Build prompt for validation round"""
        if round_num == 1:
            return f"""You are an expert bioinformatician reviewing a gene signature with {len(genes)} genes.

Based on the database analysis above, which genes (if any) should be REMOVED from this signature?

IMPORTANT:
1. The database evidence is a strong signal, but NOT absolute truth.
2. If you have strong biological reasoning to disagree with the database (e.g., context-specific expression), please do so.
3. You may also suggest removing genes NOT flagged by the database if they are biologically irrelevant.
4. DO NOT HALLUCINATE: Base all claims on verifying real biological mechanisms.

Provide:
1. List of genes to remove (if any)
2. Reasoning for each (Cite mechanism/evidence)
3. Your confidence (0-1)

Format: Gene: REASON (confidence: X)"""
        else:
            return f"""Round {round_num}: Continue the debate. Have your colleagues changed your mind? 

Provide updated recommendations with confidence scores."""
    
    def _build_expansion_prompt(self, candidate_gene: str, round_num: int) -> str:
        """Build prompt for expansion round"""
        if round_num == 1:
            return f"""You are an expert bioinformatician evaluating whether to ADD {candidate_gene} to an existing signature.

Based on the database analysis above, should we ADD this gene?

IMPORTANT:
1. Critically evaluate the database evidence.
2. If you have complementary evidence (pathway connections, co-expression) that supports addition despite weak DB signal, argue for it.
3. DO NOT HALLUCINATE: Only cite real interactions.

Vote: ADD or REJECT
Reasoning: Why?
Confidence: 0-1"""
        else:
            return f"""Round {round_num}: Continue debating {candidate_gene}. Updated vote?"""
    
    def _build_conversation_history(
        self,
        previous_rounds: List[DebateRound]
    ) -> List[Dict]:
        """Build conversation history from previous rounds"""
        history = []
        
        for debate_round in previous_rounds:
            for msg in debate_round.messages:
                role = "assistant" if msg.speaker in self.models else "user"
                history.append({
                    "role": role,
                    "content": msg.message
                })
        
        return history
    
    def _parse_removal_votes(
        self,
        messages: List[DebateMessage]
    ) -> Dict[str, float]:
        """
        Parse which genes each LLM voted to remove.
        
        Returns:
            {gene: confidence}
        """
        votes = {}
        
        for msg in messages:
            # Extract gene names mentioned
            gene_pattern = r'\b[A-Z][A-Z0-9]+\b'
            mentioned_genes = re.findall(gene_pattern, msg.message)
            
            # Check if they're recommended for removal
            lower_msg = msg.message.lower()
            if any(keyword in lower_msg for keyword in ['remove', 'exclude', 'drop', 'eliminate']):
                for gene in mentioned_genes:
                    if gene not in votes or msg.confidence > votes[gene]:
                        votes[gene] = msg.confidence or 0.5
        
        return votes
    
    def _calculate_convergence(
        self,
        messages: List[DebateMessage]
    ) -> float:
        """
        Calculate convergence rate based on agreement.
        
        Returns:
            Float 0-1 (1 = perfect agreement)
        """
        if len(messages) < 2:
            return 0.0
        
        # Simple heuristic: compare message similarity
        # In production, use semantic similarity
        
        message_texts = [msg.message.lower() for msg in messages]
        
        # Count keyword overlaps
        all_keywords = set()
        for text in message_texts:
            words = set(text.split())
            all_keywords.update(words)
        
        if not all_keywords:
            return 0.0
        
        # Calculate Jaccard similarity
        similarities = []
        for i in range(len(message_texts)):
            for j in range(i + 1, len(message_texts)):
                words_i = set(message_texts[i].split())
                words_j = set(message_texts[j].split())
                
                intersection = len(words_i & words_j)
                union = len(words_i | words_j)
                
                if union > 0:
                    similarities.append(intersection / union)
        
        if not similarities:
            return 0.0
        
        return sum(similarities) / len(similarities)
    
    def _synthesize_final_decision(
        self,
        all_rounds: List[DebateRound],
        mode: DebateMode
    ) -> Tuple[str, float]:
        """
        Meta-synthesizer: Make final decision based on all rounds.
        
        Uses weighted voting:
        - Model reliability weight
        - Confidence score
        - Recency (later rounds weighted higher)
        
        Returns:
            (decision, confidence)
        """
        if mode == DebateMode.VALIDATION:
            # Aggregate votes for gene removal
            gene_scores = {}
            
            for round_idx, debate_round in enumerate(all_rounds):
                recency_weight = 1.0 + (round_idx / len(all_rounds)) * 0.5
                
                for msg in debate_round.messages:
                    if msg.speaker not in self.models:
                        continue
                    
                    reliability = self.reliability_weights.get(msg.speaker, 1.0)
                    confidence = msg.confidence or 0.5
                    
                    # Parse genes mentioned
                    votes = self._parse_removal_votes([msg])
                    
                    for gene, vote_conf in votes.items():
                        weight = reliability * confidence * recency_weight
                        
                        if gene not in gene_scores:
                            gene_scores[gene] = 0.0
                        gene_scores[gene] += weight
            
            # Normalize scores
            if gene_scores:
                max_score = max(gene_scores.values())
                gene_scores = {g: s/max_score for g, s in gene_scores.items()}
            
            # Threshold: remove if score > 0.6
            genes_to_remove = [g for g, s in gene_scores.items() if s > 0.6]
            
            if genes_to_remove:
                avg_confidence = sum(gene_scores[g] for g in genes_to_remove) / len(genes_to_remove)
                return "remove", avg_confidence
            else:
                return "keep", 0.7
        
        else:  # EXPANSION
            # Count ADD vs REJECT votes
            add_score = 0.0
            reject_score = 0.0
            
            for round_idx, debate_round in enumerate(all_rounds):
                recency_weight = 1.0 + (round_idx / len(all_rounds)) * 0.5
                
                for msg in debate_round.messages:
                    if msg.speaker not in self.models:
                        continue
                    
                    reliability = self.reliability_weights.get(msg.speaker, 1.0)
                    confidence = msg.confidence or 0.5
                    weight = reliability * confidence * recency_weight
                    
                    lower_msg = msg.message.lower()
                    if 'add' in lower_msg or 'include' in lower_msg:
                        add_score += weight
                    elif 'reject' in lower_msg or 'exclude' in lower_msg:
                        reject_score += weight
            
            if add_score > reject_score:
                return "add", add_score / (add_score + reject_score)
            else:
                return "reject", reject_score / (add_score + reject_score)
    
    def _extract_confidence(self, text: str) -> float:
        """Extract confidence score from LLM response"""
        # Look for patterns like "confidence: 0.8" or "(0.8)"
        patterns = [
            r'confidence:\s*([0-9.]+)',
            r'\(([0-9.]+)\)',
            r'([0-9.]+)\s*confidence'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text.lower())
            if match:
                try:
                    conf = float(match.group(1))
                    if 0 <= conf <= 1:
                        return conf
                except:
                    pass
        
        return 0.5  # Default
