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
    speaker: str  # "skeptic", "discoverer", "mediator", "injector", "consensus"
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
    """Final debate outcome with publication-grade metrics"""
    mode: DebateMode
    total_rounds: int
    final_decision: str  # "remove", "add", "keep", "reject"
    affected_genes: List[str]
    confidence: float
    convergence_rate: float
    all_rounds: List[DebateRound]

    # NEW: Publication-grade quality metrics
    decision_metrics: Dict[str, float] = field(default_factory=lambda: {
        'entropy': 0.0,
        'conflict': 0.0,
        'raw_consensus': 0.0,
        'adjusted_confidence': 0.0
    })


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
        model_configs: Optional[Dict] = None,
        validate_models: bool = True,
        use_json_mode: bool = False
    ):
        """
        Args:
            api_key: Groq API key
            db_client: DatabaseClientEnhanced instance
            base_url: Groq API base URL (default: https://api.groq.com/openai/v1)
            model_configs: Optional dict of model parameters
            validate_models: Whether to validate model availability on init (default: True)
            use_json_mode: Use structured JSON outputs for better parsing (default: False)
        """
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.db_client = db_client
        self.injector = DatabaseInjector(db_client)
        self.use_json_mode = use_json_mode

        # ADVERSARIAL ROLE ASSIGNMENT (ARA) - Conservative Configuration
        # Uses Llama + Gemma (2 families instead of 3, but all verified working)
        self.model_roles = {
            "skeptic": {
                "id": "llama-3.3-70b-versatile",  # Meta AI - Most capable
                "company": "Meta",
                "role_name": "SKEPTIC",
                "system_prompt": (
                    "You are Reviewer #3 - the conservative skeptic in this biological debate. "
                    "Your PRIMARY directive: DISTRUST LLM intuition and TRUST database evidence. "
                    "You actively look for false positives and require overwhelming evidence to approve genes. "
                    "Be critical, demand citations, and reject unless database strongly supports."
                )
            },
            "discoverer": {
                "id": "llama-3.1-8b-instant",  # Meta AI - Fast and creative
                "company": "Meta",
                "role_name": "DISCOVERER",
                "system_prompt": (
                    "You are the hypothesis generator - the creative discoverer in this biological debate. "
                    "Your PRIMARY directive: EXPLORE novel connections not yet in databases. "
                    "If there is ANY plausible biological mechanism or literature support, argue for it. "
                    "Be open-minded, consider emerging evidence, and champion hidden gems."
                )
            },
            "mediator": {
                "id": "gemma2-9b-it",  # Google - Different architecture
                "company": "Google",
                "role_name": "MEDIATOR",
                "system_prompt": (
                    "You are the chairperson - the balanced mediator in this biological debate. "
                    "Your PRIMARY directive: Find CONSENSUS between the Skeptic and Discoverer. "
                    "Weigh BOTH database evidence AND biological plausibility equally. "
                    "Be fair, balanced, and provide the tiebreaker perspective."
                )
            }
        }

        # Backward compatibility: Extract model IDs
        self.models = {k: v["id"] for k, v in self.model_roles.items()}

        # Model reliability weights (adjusted for configuration)
        self.reliability_weights = {
            "skeptic": 1.2,    # Highest weight - most capable model
            "discoverer": 1.0,  # Medium weight - creative but needs validation
            "mediator": 1.0     # Medium weight - tiebreaker
        }

        # Override with custom configs if provided
        if model_configs:
            for role, model_id in model_configs.items():
                if role in self.model_roles:
                    self.model_roles[role]["id"] = model_id
                    self.models[role] = model_id

        # ✅ FIX BUG #2: Validate model availability
        if validate_models:
            self._validate_model_availability()

    def _validate_model_availability(self):
        """
        Validate that configured models exist in Groq.

        Prints warnings for unavailable models but doesn't block initialization.
        This allows the system to start even if some models are unavailable.
        """
        try:
            # Note: models.list() is synchronous in the OpenAI SDK
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            # Create a temporary sync client for validation
            from openai import OpenAI
            sync_client = OpenAI(
                api_key=self.client.api_key,
                base_url=self.client.base_url
            )

            models_response = sync_client.models.list()
            available_model_ids = {model.id for model in models_response.data}

            print("🔍 Validating Groq models...")
            print(f"   Found {len(available_model_ids)} available models")

            # Check each configured model
            unavailable = []
            for role, model_id in self.models.items():
                if model_id in available_model_ids:
                    print(f"   ✅ {role}: {model_id}")
                else:
                    print(f"   ❌ {role}: {model_id} (NOT FOUND)")
                    unavailable.append((role, model_id))

            # Suggest alternatives for unavailable models
            if unavailable:
                print("\n⚠️  WARNING: Some models are unavailable!")
                print("Available Groq models:")
                for model_id in sorted(available_model_ids):
                    print(f"   • {model_id}")

                print("\n💡 Suggested fixes:")
                for role, model_id in unavailable:
                    # Find similar models
                    if 'gemma' in model_id.lower():
                        alternatives = [m for m in available_model_ids if 'gemma' in m.lower()]
                    elif 'llama' in model_id.lower():
                        alternatives = [m for m in available_model_ids if 'llama' in m.lower()]
                    else:
                        alternatives = list(available_model_ids)[:3]

                    if alternatives:
                        print(f"   {role}: Try '{alternatives[0]}' instead of '{model_id}'")

                print("\nDebates may fail with unavailable models. Update model_configs or use:")
                print("  python groq_model_diagnostic.py <your_api_key>")

        except Exception as e:
            print(f"⚠️  Could not validate models: {e}")
            print("Continuing without validation...")

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
        
        # Meta-synthesizer makes final decision with metrics
        self._last_decision_metrics = {}  # Initialize
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
            all_rounds=all_rounds,
            decision_metrics=getattr(self, '_last_decision_metrics', {})
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
        
        # Final decision with metrics
        self._last_decision_metrics = {}  # Initialize
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
            all_rounds=all_rounds,
            decision_metrics=getattr(self, '_last_decision_metrics', {})
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
        # 1. Prepare CURRENT round messages (Injector + Prompt)
        # We don't build full history here yet, as it needs to be role-specific
        current_round_msgs = []

        if injector_context:
            current_round_msgs.append({
                "role": "user",
                "content": injector_context.message
            })

        prompt = self._build_validation_prompt(genes, round_num)
        current_round_msgs.append({
            "role": "user",
            "content": prompt
        })

        # 2. Query all LLMs in parallel with role-specific system prompts
        tasks = []
        for model_name, model_id in self.models.items():
            # Build role-specific history (System Prompt + Previous Rounds)
            role_history = self._build_conversation_history_with_roles(
                previous_rounds, model_name
            )
            # Add current round context
            role_history.extend(current_round_msgs)  # ✅ FIXED: Uses current_round_msgs

            tasks.append(
                self._query_llm_async(model_name, model_id, role_history, use_json=self.use_json_mode)
            )
        
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
        # 1. Prepare CURRENT round messages
        current_round_msgs = []

        if injector_context:
            current_round_msgs.append({
                "role": "user",
                "content": injector_context.message
            })

        prompt = self._build_expansion_prompt(candidate_gene, round_num)
        current_round_msgs.append({
            "role": "user",
            "content": prompt
        })

        # 2. Query all LLMs in parallel with role-specific system prompts
        tasks = []
        for model_name, model_id in self.models.items():
            # Build role-specific history
            role_history = self._build_conversation_history_with_roles(
                previous_rounds, model_name
            )
            # Add current round context
            role_history.extend(current_round_msgs)  # ✅ FIXED: Uses current_round_msgs

            tasks.append(
                self._query_llm_async(model_name, model_id, role_history, use_json=self.use_json_mode)
            )

        responses = await asyncio.gather(*tasks, return_exceptions=True)

        messages = []
        for model_name, response in zip(self.models.keys(), responses):
            if isinstance(response, Exception):
                print(f"⚠️ {model_name} failed in expansion debate: {response}")
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
        conversation_history: List[Dict],
        use_json: bool = False
    ) -> str:
        """
        Async LLM query via Groq API (OpenAI-compatible).

        Args:
            model_name: Human-readable model name (skeptic, discoverer, mediator)
            model_id: Groq model ID
            conversation_history: Chat messages
            use_json: Force JSON output format (requires JSON prompt)

        Raises:
            RuntimeError: If the model query fails
        """
        try:
            # Prepare API call parameters
            api_params = {
                "model": model_id,
                "messages": conversation_history,
                "max_tokens": 500,
                "temperature": 0.7
            }

            # Add JSON mode if enabled (requires supported models)
            if use_json:
                api_params["response_format"] = {"type": "json_object"}

            # Call Groq API via AsyncOpenAI client
            response = await self.client.chat.completions.create(**api_params)

            return response.choices[0].message.content
        except Exception as e:
            # ✅ FIX: Raise exception instead of returning error string
            # This ensures proper error detection in the calling code
            raise RuntimeError(f"Model {model_name} ({model_id}) failed: {str(e)}")
    
    def _build_validation_prompt(self, genes: List[str], round_num: int) -> str:
        """Build prompt for validation round"""
        if self.use_json_mode:
            # JSON mode prompt
            if round_num == 1:
                return f"""You are an expert bioinformatician reviewing a gene signature with {len(genes)} genes.

Based on the database analysis above, which genes (if any) should be REMOVED from this signature?

IMPORTANT:
1. The database evidence is a strong signal, but NOT absolute truth.
2. If you have strong biological reasoning to disagree with the database, please do so.
3. You may also suggest removing genes NOT flagged by the database if biologically irrelevant.
4. DO NOT HALLUCINATE: Base all claims on real biological mechanisms.

Respond with JSON ONLY in this exact format:
{{
  "genes_to_remove": ["GENE1", "GENE2"],
  "reasoning": {{
    "GENE1": "specific biological reason",
    "GENE2": "specific biological reason"
  }},
  "confidence": 0.85
}}

If no genes should be removed, use empty array: {{"genes_to_remove": [], "reasoning": {{}}, "confidence": 0.9}}"""
            else:
                return f"""Round {round_num}: Continue the debate. Have your colleagues changed your mind?

Provide updated recommendations in JSON format:
{{
  "genes_to_remove": ["GENE1", "GENE2"],
  "reasoning": {{"GENE1": "reason", "GENE2": "reason"}},
  "confidence": 0.85
}}"""
        else:
            # Original text mode
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

    def _build_conversation_history_with_roles(
        self,
        previous_rounds: List[DebateRound],
        current_role: str
    ) -> List[Dict]:
        """
        Build conversation history with role-specific system prompt.

        Args:
            previous_rounds: Previous debate rounds
            current_role: 'skeptic', 'discoverer', or 'mediator'

        Returns:
            Conversation history with role-aware system prompt
        """
        history = []

        # Add role-specific system prompt at the beginning
        if current_role in self.model_roles:
            role_info = self.model_roles[current_role]
            history.append({
                "role": "system",
                "content": role_info["system_prompt"]
            })

        # Add previous rounds
        for debate_round in previous_rounds:
            for msg in debate_round.messages:
                role = "assistant" if msg.speaker in self.models else "user"
                history.append({
                    "role": role,
                    "content": msg.message
                })

        return history

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

        Supports both JSON and text formats.

        Returns:
            {gene: confidence}
        """
        votes = {}

        for msg in messages:
            # Try JSON parsing first if JSON mode is enabled
            if self.use_json_mode:
                try:
                    import json
                    data = json.loads(msg.message)

                    genes_to_remove = data.get("genes_to_remove", [])
                    confidence = data.get("confidence", msg.confidence or 0.5)

                    for gene in genes_to_remove:
                        if gene not in votes or confidence > votes[gene]:
                            votes[gene] = confidence

                    continue  # Skip text parsing if JSON worked
                except (json.JSONDecodeError, KeyError, TypeError):
                    # Fall back to text parsing
                    pass

            # Text-based parsing (original logic)
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

        In JSON mode: Uses gene set overlap (more accurate)
        In text mode: Uses word overlap (original heuristic)

        Returns:
            Float 0-1 (1 = perfect agreement)
        """
        if len(messages) < 2:
            return 0.0

        # ✅ IMPROVED: JSON mode uses gene-based convergence
        if self.use_json_mode:
            import json
            gene_sets = []

            for msg in messages:
                try:
                    data = json.loads(msg.message)
                    genes = set(data.get("genes_to_remove", []))
                    gene_sets.append(genes)
                except (json.JSONDecodeError, KeyError, TypeError):
                    # Fall back to text parsing
                    gene_pattern = r'\b[A-Z][A-Z0-9]+\b'
                    genes = set(re.findall(gene_pattern, msg.message))
                    gene_sets.append(genes)

            # Calculate pairwise Jaccard similarity of gene sets
            similarities = []
            for i in range(len(gene_sets)):
                for j in range(i + 1, len(gene_sets)):
                    set_i = gene_sets[i]
                    set_j = gene_sets[j]

                    if len(set_i) == 0 and len(set_j) == 0:
                        # Both agree on removing nothing
                        similarities.append(1.0)
                    elif len(set_i) == 0 or len(set_j) == 0:
                        # One wants to remove genes, other doesn't
                        similarities.append(0.0)
                    else:
                        intersection = len(set_i & set_j)
                        union = len(set_i | set_j)
                        similarities.append(intersection / union if union > 0 else 0.0)

            return sum(similarities) / len(similarities) if similarities else 0.0

        # Original text-based convergence
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
        PUBLICATION-GRADE: Entropy-adjusted consensus with conflict discounting.

        Mathematical Model:
        1. Shannon Entropy: Measures model uncertainty
        2. Vote Variance: Measures inter-model conflict
        3. Final Score: Weighted consensus discounted by uncertainty + conflict

        Novel Contribution: Quantifies "hallucination gap" when LLMs ignore database.

        Returns:
            (decision, confidence)
        """
        import numpy as np

        if mode == DebateMode.VALIDATION:
            # Aggregate votes for gene removal across ALL rounds
            gene_scores = {}

            for round_idx, debate_round in enumerate(all_rounds):
                # Recency weight: Later rounds matter more (converged opinions)
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

            # Extract final round for entropy calculation
            last_round = all_rounds[-1]
            confidences = [
                (msg.confidence or 0.5) * self.reliability_weights.get(msg.speaker, 1.0)
                for msg in last_round.messages
                if msg.speaker in self.models
            ]

            # METRIC 1: Shannon Entropy (Uncertainty)
            if confidences:
                probs = np.array(confidences) / np.sum(confidences)
                entropy = -np.sum(probs * np.log2(probs + 1e-10))
                max_entropy = np.log2(len(confidences))
                normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
            else:
                normalized_entropy = 1.0  # Maximum uncertainty

            # METRIC 2: Vote Variance (Conflict)
            # Extract binary votes (1=remove, 0=keep) from last round
            binary_votes = [
                1.0 if self._parse_removal_votes([msg]) else 0.0
                for msg in last_round.messages
                if msg.speaker in self.models
            ]

            if len(binary_votes) > 1:
                vote_variance = np.var(binary_votes)
                conflict_penalty = 1.0 - (vote_variance * 4)  # Scale to [0,1]
            else:
                conflict_penalty = 1.0

            # Normalize gene scores
            if gene_scores:
                max_score = max(gene_scores.values())
                gene_scores = {g: s/max_score for g, s in gene_scores.items()}

            # Threshold: remove if score > 0.6, but discount by uncertainty
            genes_to_remove = [
                g for g, s in gene_scores.items()
                if s > 0.6
            ]

            if genes_to_remove:
                # Average confidence of removal votes
                avg_confidence = sum(gene_scores[g] for g in genes_to_remove) / len(genes_to_remove)

                # Apply uncertainty discount
                final_confidence = avg_confidence * conflict_penalty * (1.0 - normalized_entropy * 0.3)

                # Store metrics for export (PUBLICATION DATA)
                self._last_decision_metrics = {
                    'entropy': float(normalized_entropy),
                    'conflict': float(vote_variance) if len(binary_votes) > 1 else 0.0,
                    'raw_consensus': float(avg_confidence),
                    'adjusted_confidence': float(final_confidence)
                }

                return "remove", final_confidence
            else:
                self._last_decision_metrics = {
                    'entropy': float(normalized_entropy),
                    'conflict': float(vote_variance) if len(binary_votes) > 1 else 0.0,
                    'raw_consensus': 0.7,
                    'adjusted_confidence': 0.7
                }
                return "keep", 0.7

        else:  # EXPANSION mode
            # Extract final round data
            last_round = all_rounds[-1]

            confidences = []
            votes = []  # 1=add, 0=reject

            for msg in last_round.messages:
                if msg.speaker not in self.models:
                    continue

                reliability = self.reliability_weights.get(msg.speaker, 1.0)
                conf = (msg.confidence or 0.5) * reliability
                confidences.append(conf)

                # Binary vote
                lower_msg = msg.message.lower()
                vote = 1.0 if ('add' in lower_msg or 'include' in lower_msg) else 0.0
                votes.append(vote)

            if not confidences:
                self._last_decision_metrics = {
                    'entropy': 1.0,
                    'conflict': 1.0,
                    'raw_consensus': 0.5,
                    'adjusted_confidence': 0.5
                }
                return "reject", 0.5

            # METRIC 1: Shannon Entropy
            probs = np.array(confidences) / np.sum(confidences)
            entropy = -np.sum(probs * np.log2(probs + 1e-10))
            max_entropy = np.log2(len(confidences))
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

            # METRIC 2: Vote Variance
            vote_variance = np.var(votes) if len(votes) > 1 else 0.0
            conflict_penalty = 1.0 - (vote_variance * 4)

            # Weighted consensus
            weighted_vote = np.average(votes, weights=confidences)

            # Final score with uncertainty discount
            final_confidence = weighted_vote * conflict_penalty * (1.0 - normalized_entropy * 0.3)

            # Store metrics
            self._last_decision_metrics = {
                'entropy': float(normalized_entropy),
                'conflict': float(vote_variance),
                'raw_consensus': float(weighted_vote),
                'adjusted_confidence': float(final_confidence)
            }

            # Decision
            if final_confidence > 0.6:
                return "add", final_confidence
            else:
                return "reject", 1.0 - final_confidence
    
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
