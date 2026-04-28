"""
CAAAPT Attributed Analyzer
RAG + CoT based attack attribution analyzer for APT detection
Implements the backend LLM analysis pipeline as described in Section 3.2

Core components:
- Retrieval-Augmented Generation (RAG) for knowledge injection
- Four-stage Chain-of-Thought (CoT) reasoning
- Confidence-based decision making with manual review delegation

Sanitized version for submission - No hardcoded API keys, internal IPs, or sensitive endpoints
"""

import os
import json
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

# Import internal modules
from retriever import HybridRetriever, RetrievalResult, create_retriever
from cot_prompts import ChainOfThoughtReasoner, CoTPromptTemplates
from llm_client import LLMClient, APTAttributionLLM, create_llm_client_from_env, ModelProvider


class AttributionDecision(Enum):
    """Attribution decision types"""
    ACCEPT = "accept"
    MANUAL_REVIEW = "manual_review"
    REJECT = "reject"
    DEFER = "defer"


@dataclass
class AttributedResult:
    """
    Complete attribution result structure
    Corresponds to Equation (4) in the paper:
    F_back(z) = {(C_final, R_report) | C_final >= tau_back}
    """
    sample_id: str
    timestamp: str
    decision: AttributionDecision
    final_confidence: float
    report: Dict[str, Any]
    stage_outputs: List[Dict[str, Any]]
    tau_back_used: float
    latency_ms: float
    retrieval_context: Dict[str, Any]

    def to_json(self) -> str:
        """Serialize to JSON string"""
        return json.dumps({
            "sample_id": self.sample_id,
            "timestamp": self.timestamp,
            "decision": self.decision.value,
            "final_confidence": self.final_confidence,
            "report": self.report,
            "stage_outputs": self.stage_outputs,
            "tau_back_used": self.tau_back_used,
            "latency_ms": self.latency_ms
        }, indent=2)

    def to_auditable_trace(self) -> str:
        """Generate human-readable auditable trace"""
        trace = []
        trace.append("=" * 80)
        trace.append(f"CAAAPT Attribution Report - Sample: {self.sample_id}")
        trace.append(f"Timestamp: {self.timestamp}")
        trace.append(f"Decision: {self.decision.value} (tau_back={self.tau_back_used})")
        trace.append(f"Final Confidence: {self.final_confidence:.4f}")
        trace.append(f"Latency: {self.latency_ms:.2f} ms")
        trace.append("=" * 80)

        for stage in self.stage_outputs:
            trace.append(f"\n--- {stage.get('stage_name', 'unknown').upper()} ---")
            trace.append(f"Confidence: {stage.get('confidence', 0)}")
            output = stage.get('output', {})
            if isinstance(output, dict):
                if 'summary' in output:
                    trace.append(f"Summary: {output['summary']}")
                if 'attack_narrative' in output:
                    trace.append(f"Narrative: {output['attack_narrative']}")
                if 'decision' in output:
                    trace.append(f"Stage Decision: {output['decision']}")
            trace.append("-" * 40)

        trace.append("\n" + "=" * 80)
        return "\n".join(trace)


class AttributedAnalyzer:
    """
    RAG + CoT based attack attribution analyzer

    Implements the backend processing pipeline described in Section 3.2:
    1. Retrieval-Augmented Generation (RAG) for knowledge context
    2. Four-stage Chain-of-Thought reasoning
    3. Comprehensive confidence evaluation
    4. Manual review delegation for low-confidence samples
    """

    def __init__(self,
                 retriever: Optional[HybridRetriever] = None,
                 llm_client: Optional[LLMClient] = None,
                 tau_back: float = 0.75,
                 knowledge_base_path: str = "data/knowledge_base",
                 use_cache: bool = True):
        """
        Initialize the attributed analyzer

        Args:
            retriever: Hybrid retriever instance (created if None)
            llm_client: LLM client instance (created from env if None)
            tau_back: Backend confidence threshold for attribution acceptance
            knowledge_base_path: Path to knowledge base
            use_cache: Whether to use caching for retrieval
        """
        self.tau_back = tau_back
        self.use_cache = use_cache
        self.knowledge_base_path = knowledge_base_path
        self.cache: Dict[str, AttributedResult] = {}

        # Initialize retriever
        if retriever:
            self.retriever = retriever
        else:
            self.retriever = create_retriever(
                structured_db_path=os.path.join(knowledge_base_path, "structured.db"),
                vector_db_path=os.path.join(knowledge_base_path, "vectors.pkl")
            )

        # Initialize LLM
        if llm_client:
            self.llm_client = llm_client
            self.apt_llm = APTAttributionLLM(llm_client)
        else:
            self.llm_client = create_llm_client_from_env()
            self.apt_llm = APTAttributionLLM(self.llm_client)

        # Initialize CoT reasoner
        self.cot_reasoner = ChainOfThoughtReasoner(llm_client=self.llm_client)

        # Prompt templates
        self.prompts = CoTPromptTemplates()

        # Statistics
        self.stats = {
            "total_processed": 0,
            "accepted": 0,
            "manual_review": 0,
            "rejected": 0,
            "average_confidence": 0.0,
            "average_latency_ms": 0.0
        }

        print(f"[INFO] AttributedAnalyzer initialized with tau_back={tau_back}")

    def _get_cache_key(self, sample: Dict[str, Any]) -> str:
        """Generate cache key for a sample"""
        sample_str = json.dumps(sample, sort_keys=True)
        return hashlib.md5(sample_str.encode()).hexdigest()

    def _retrieve_knowledge_context(self,
                                    event_sequence: str,
                                    tactics: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Retrieve relevant knowledge for RAG context

        As described in Section 3.2.2:
        Multi-modal fusion retrieval mechanism combining structured queries
        and semantic vector retrieval.
        """
        context = {
            "structured_knowledge": [],
            "semantic_knowledge": [],
            "related_iocs": [],
            "attack_patterns": []
        }

        # Default tactics if none specified
        if tactics is None:
            tactics = ["Execution", "Defense Evasion", "Persistence"]

        # Hybrid retrieval for each tactic
        for tactic in tactics:
            results = self.retriever.hybrid_retrieval(
                query=event_sequence,
                tactic_filter=tactic,
                top_k=5,
                alpha=0.6,
                use_cache=self.use_cache
            )

            for r in results:
                entry = {
                    "content": r.content,
                    "tactic": r.tactic,
                    "technique": r.technique,
                    "score": r.score,
                    "source": r.source
                }

                if r.source.startswith("hybrid") or r.source == "structured_sql":
                    context["structured_knowledge"].append(entry)
                else:
                    context["semantic_knowledge"].append(entry)

        # Deduplicate
        context["structured_knowledge"] = list({
                                                   f"{e['tactic']}_{e['technique']}": e
                                                   for e in context["structured_knowledge"]
                                               }.values())

        context["semantic_knowledge"] = list({
                                                 e["content"][:100]: e
                                                 for e in context["semantic_knowledge"]
                                             }.values())

        return context

    def _format_knowledge_context(self, knowledge_context: Dict[str, Any]) -> str:
        """Format knowledge context for LLM prompt"""
        formatted_parts = []

        if knowledge_context.get("structured_knowledge"):
            formatted_parts.append("=== Structured Knowledge (ATT&CK Tactics/Techniques) ===")
            for k in knowledge_context["structured_knowledge"][:10]:
                formatted_parts.append(
                    f"- [{k.get('tactic', 'unknown')}] {k.get('technique', '')}: {k['content'][:200]}")

        if knowledge_context.get("semantic_knowledge"):
            formatted_parts.append("\n=== Semantic Knowledge (Threat Intelligence) ===")
            for k in knowledge_context["semantic_knowledge"][:5]:
                formatted_parts.append(
                    f"- [{k.get('tactic', 'unknown')}] Score={k.get('score', 0):.3f}: {k['content'][:150]}...")

        if knowledge_context.get("related_iocs"):
            formatted_parts.append("\n=== Related IOCs ===")
            for ioc in knowledge_context["related_iocs"][:5]:
                formatted_parts.append(f"- {ioc}")

        return "\n".join(formatted_parts) if formatted_parts else "No relevant knowledge retrieved."

    def _format_event_sequence(self, event_sequence: Dict[str, Any]) -> str:
        """Format event sequence for LLM prompt"""
        if isinstance(event_sequence, str):
            return event_sequence

        if isinstance(event_sequence, list):
            lines = []
            for i, event in enumerate(event_sequence[:20]):
                if isinstance(event, dict):
                    lines.append(
                        f"[{i + 1}] Time={event.get('timestamp', 'N/A')} Type={event.get('event_type', 'unknown')} "
                        f"Source={event.get('source_entity', 'N/A')} Target={event.get('target_entity', 'N/A')}")
                else:
                    lines.append(f"[{i + 1}] {event}")
            return "\n".join(lines)

        return str(event_sequence)

    def analyze(self,
                event_sequence: Dict[str, Any],
                xgboost_score: float,
                sample_id: Optional[str] = None,
                override_tau_back: Optional[float] = None,
                use_cache: bool = True) -> AttributedResult:
        """
        Perform RAG + CoT based attribution analysis

        This is the main entry point for backend attribution as described in Algorithm 2.

        Args:
            event_sequence: Suspicious event sequence to analyze
            xgboost_score: Anomaly score from frontend XGBoost (C_d)
            sample_id: Optional sample identifier
            override_tau_back: Override the default tau_back threshold
            use_cache: Whether to use cached results

        Returns:
            AttributedResult containing the analysis outcome
        """
        import time
        start_time = time.time()

        # Generate sample ID if not provided
        if sample_id is None:
            sample_id = hashlib.md5(json.dumps(event_sequence, sort_keys=True).encode()).hexdigest()[:16]

        # Check cache
        cache_key = self._get_cache_key({"seq": event_sequence, "score": xgboost_score})
        if use_cache and cache_key in self.cache:
            cached_result = self.cache[cache_key]
            if (datetime.now().timestamp() - datetime.fromisoformat(cached_result.timestamp).timestamp()) < 3600:
                return cached_result

        # Use provided tau_back or default
        tau_back = override_tau_back if override_tau_back is not None else self.tau_back

        # Step 1: Retrieve knowledge context (RAG)
        event_sequence_str = self._format_event_sequence(event_sequence)
        knowledge_context = self._retrieve_knowledge_context(event_sequence_str)
        knowledge_context_str = self._format_knowledge_context(knowledge_context)

        # Step 2: Run four-stage CoT reasoning
        stage_outputs = []

        # Stage 1: Event Semantic Reconstruction
        reconstruction = self.cot_reasoner.stage1_reconstruct(
            event_sequence=event_sequence_str,
            knowledge_context=knowledge_context_str
        )
        stage_outputs.append({
            "stage_name": "reconstruction",
            "confidence": reconstruction.get("reconstruction_confidence", 0.5),
            "output": reconstruction
        })

        # Stage 2: Tactical Ontology Alignment
        alignment = self.cot_reasoner.stage2_align(
            behavior_sequence=json.dumps(reconstruction.get("behavior_sequence", []), indent=2),
            causal_graph=json.dumps(reconstruction.get("causal_graph", {}), indent=2),
            knowledge_context=knowledge_context_str
        )
        stage_outputs.append({
            "stage_name": "alignment",
            "confidence": alignment.get("alignment_confidence", 0.5),
            "output": alignment
        })

        # Stage 3: Attack Intention Inference
        intention = self.cot_reasoner.stage3_infer(
            tactic_sequence=json.dumps(alignment.get("tactic_sequence", []), indent=2),
            behavior_sequence=json.dumps(reconstruction.get("behavior_sequence", []), indent=2)
        )
        stage_outputs.append({
            "stage_name": "intention",
            "confidence": intention.get("intention_confidence", 0.5),
            "output": intention
        })

        # Stage 4: Comprehensive Confidence Evaluation
        confidence_eval = self.cot_reasoner.stage4_evaluate(
            reconstruction_output=reconstruction,
            alignment_output=alignment,
            intention_output=intention,
            xgboost_score=xgboost_score
        )
        stage_outputs.append({
            "stage_name": "confidence",
            "confidence": confidence_eval.get("final_confidence", 0.5),
            "output": confidence_eval
        })

        # Step 3: Extract final confidence and decision
        final_confidence = confidence_eval.get("final_confidence", 0.5)

        if final_confidence >= tau_back:
            decision = AttributionDecision.ACCEPT
        else:
            decision = AttributionDecision.MANUAL_REVIEW

        # Step 4: Build attribution report
        report = {
            "summary": reconstruction.get("summary", ""),
            "tactic_sequence": alignment.get("tactic_sequence", []),
            "attack_narrative": intention.get("attack_narrative", ""),
            "suspected_objectives": intention.get("suspected_objectives", []),
            "campaign_attribution": intention.get("campaign_attribution", {}),
            "final_confidence": final_confidence,
            "decision": decision.value,
            "tau_back_used": tau_back
        }

        # Step 5: Create result
        latency_ms = (time.time() - start_time) * 1000

        result = AttributedResult(
            sample_id=sample_id,
            timestamp=datetime.now().isoformat(),
            decision=decision,
            final_confidence=final_confidence,
            report=report,
            stage_outputs=stage_outputs,
            tau_back_used=tau_back,
            latency_ms=latency_ms,
            retrieval_context=knowledge_context
        )

        # Update cache
        if use_cache:
            self.cache[cache_key] = result

        # Update statistics
        self._update_stats(result)

        return result

    def _update_stats(self, result: AttributedResult):
        """Update internal statistics"""
        self.stats["total_processed"] += 1

        if result.decision == AttributionDecision.ACCEPT:
            self.stats["accepted"] += 1
        elif result.decision == AttributionDecision.MANUAL_REVIEW:
            self.stats["manual_review"] += 1
        else:
            self.stats["rejected"] += 1

        # Update running average
        n = self.stats["total_processed"]
        self.stats["average_confidence"] = (
                (self.stats["average_confidence"] * (n - 1) + result.final_confidence) / n
        )
        self.stats["average_latency_ms"] = (
                (self.stats["average_latency_ms"] * (n - 1) + result.latency_ms) / n
        )

    def analyze_batch(self,
                      samples: List[Tuple[Dict[str, Any], float]],
                      sample_ids: Optional[List[str]] = None) -> List[AttributedResult]:
        """
        Analyze a batch of samples

        Args:
            samples: List of (event_sequence, xgboost_score) tuples
            sample_ids: Optional list of sample IDs

        Returns:
            List of AttributedResult objects
        """
        results = []

        for i, (event_sequence, xgboost_score) in enumerate(samples):
            sample_id = sample_ids[i] if sample_ids and i < len(sample_ids) else None
            result = self.analyze(event_sequence, xgboost_score, sample_id)
            results.append(result)

            # Print progress for long batches
            if (i + 1) % 10 == 0:
                print(f"[INFO] Processed {i + 1}/{len(samples)} samples")

        return results

    def get_samples_for_manual_review(self) -> List[AttributedResult]:
        """
        Get samples that were delegated to manual review
        These are samples with C_final < tau_back as per Equation (4)
        """
        return [r for r in self.cache.values() if r.decision == AttributionDecision.MANUAL_REVIEW]

    def manual_review_override(self,
                               sample_id: str,
                               override_decision: AttributionDecision,
                               reviewer_notes: str) -> Optional[AttributedResult]:
        """
        Override attribution decision after manual review

        Args:
            sample_id: ID of the sample to override
            override_decision: New decision (ACCEPT or REJECT)
            reviewer_notes: Notes from human reviewer

        Returns:
            Updated AttributedResult or None if sample not found
        """
        # Find the result in cache
        cached_result = None
        for key, result in self.cache.items():
            if result.sample_id == sample_id:
                cached_result = result
                break

        if cached_result is None:
            print(f"[WARN] Sample {sample_id} not found in cache")
            return None

        # Update result with override
        updated_result = AttributedResult(
            sample_id=cached_result.sample_id,
            timestamp=datetime.now().isoformat(),
            decision=override_decision,
            final_confidence=cached_result.final_confidence,
            report={
                **cached_result.report,
                "manual_review_notes": reviewer_notes,
                "original_decision": cached_result.decision.value
            },
            stage_outputs=cached_result.stage_outputs,
            tau_back_used=cached_result.tau_back_used,
            latency_ms=cached_result.latency_ms,
            retrieval_context=cached_result.retrieval_context
        )

        # Update cache
        for key in list(self.cache.keys()):
            if self.cache[key].sample_id == sample_id:
                self.cache[key] = updated_result
                break

        # Update statistics
        if cached_result.decision != override_decision:
            if override_decision == AttributionDecision.ACCEPT:
                self.stats["accepted"] += 1
                self.stats["manual_review"] -= 1
            elif override_decision == AttributionDecision.REJECT:
                self.stats["rejected"] += 1
                self.stats["manual_review"] -= 1

        return updated_result

    def get_statistics(self) -> Dict[str, Any]:
        """Get analyzer statistics"""
        return {
            **self.stats,
            "cache_size": len(self.cache),
            "tau_back": self.tau_back,
            "llm_cost": self.llm_client.get_cost_estimate() if hasattr(self.llm_client, 'get_cost_estimate') else {}
        }

    def reset(self):
        """Reset analyzer state (clear cache and statistics)"""
        self.cache.clear()
        self.stats = {
            "total_processed": 0,
            "accepted": 0,
            "manual_review": 0,
            "rejected": 0,
            "average_confidence": 0.0,
            "average_latency_ms": 0.0
        }
        if hasattr(self.llm_client, 'reset_stats'):
            self.llm_client.reset_stats()

    def close(self):
        """Close resources"""
        if hasattr(self.retriever, 'close'):
            self.retriever.close()


class ManualReviewHandler:
    """
    Handler for manual review delegation

    As described in Section 3.2 and Equation (4):
    If C_final < tau_back, the sample triggers manual review.
    This combines LLM automation with human expertise to control
    false positives and reduce labor costs.
    """

    def __init__(self, analyzer: AttributedAnalyzer):
        self.analyzer = analyzer
        self.review_queue: List[AttributedResult] = []
        self.review_history: List[Dict[str, Any]] = []

    def refresh_queue(self):
        """Refresh the manual review queue from analyzer cache"""
        self.review_queue = self.analyzer.get_samples_for_manual_review()
        return len(self.review_queue)

    def get_next_review(self) -> Optional[AttributedResult]:
        """Get the next sample awaiting review"""
        if not self.review_queue:
            self.refresh_queue()

        if self.review_queue:
            return self.review_queue.pop(0)
        return None

    def submit_review(self,
                      sample_id: str,
                      decision: AttributionDecision,
                      notes: str,
                      reviewer_id: str = "human_analyst") -> bool:
        """
        Submit manual review decision

        Args:
            sample_id: ID of the reviewed sample
            decision: ACCEPT or REJECT (MANUAL_REVIEW not allowed)
            notes: Reviewer notes
            reviewer_id: Identifier of the reviewer

        Returns:
            True if update was successful
        """
        if decision == AttributionDecision.MANUAL_REVIEW:
            raise ValueError("Manual review decision cannot be MANUAL_REVIEW")

        result = self.analyzer.manual_review_override(sample_id, decision, notes)

        if result:
            self.review_history.append({
                "sample_id": sample_id,
                "decision": decision.value,
                "notes": notes,
                "reviewer_id": reviewer_id,
                "timestamp": datetime.now().isoformat()
            })
            return True

        return False

    def get_review_statistics(self) -> Dict[str, Any]:
        """Get statistics about manual reviews"""
        total_reviews = len(self.review_history)
        if total_reviews == 0:
            return {"total_reviews": 0, "accept_rate": 0.0}

        accepted = sum(1 for r in self.review_history if r["decision"] == "accept")

        return {
            "total_reviews": total_reviews,
            "accepted": accepted,
            "rejected": total_reviews - accepted,
            "accept_rate": accepted / total_reviews
        }


# ================================================================
# Factory Functions and Testing
# ================================================================

def create_analyzer_from_config(config_path: Optional[str] = None) -> AttributedAnalyzer:
    """
    Create AttributedAnalyzer from configuration file

    Expected config format (YAML or JSON):
    {
        "tau_back": 0.75,
        "knowledge_base_path": "data/knowledge_base",
        "llm": {
            "provider": "deepseek_v2",
            "model": "deepseek-chat",
            "temperature": 0.1
        },
        "retrieval": {
            "alpha": 0.6,
            "top_k": 10,
            "use_cache": true
        }
    }
    """
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            if config_path.endswith('.json'):
                config = json.load(f)
            else:
                # Assume YAML if not JSON
                try:
                    import yaml
                    config = yaml.safe_load(f)
                except ImportError:
                    print("[WARN] PyYAML not available, using defaults")
                    config = {}
    else:
        config = {}

    tau_back = config.get("tau_back", 0.75)
    knowledge_base_path = config.get("knowledge_base_path", "data/knowledge_base")
    use_cache = config.get("retrieval", {}).get("use_cache", True)

    # Create retriever
    retriever = create_retriever(
        structured_db_path=os.path.join(knowledge_base_path, "structured.db"),
        vector_db_path=os.path.join(knowledge_base_path, "vectors.pkl")
    )

    # Optional: Configure retriever alpha
    if hasattr(retriever, 'alpha'):
        retriever.alpha = config.get("retrieval", {}).get("alpha", 0.6)

    # Create LLM client
    llm_client = create_llm_client_from_env()

    return AttributedAnalyzer(
        retriever=retriever,
        llm_client=llm_client,
        tau_back=tau_back,
        knowledge_base_path=knowledge_base_path,
        use_cache=use_cache
    )


def test_analyzer():
    """Test the attributed analyzer"""
    print("=" * 60)
    print("Testing CAAAPT Attributed Analyzer")
    print("=" * 60)

    # Create analyzer in test mode
    # Note: Use mock LLM for testing without API key
    from llm_client import LLMClient, ModelProvider

    mock_llm = LLMClient(provider=ModelProvider.MOCK)

    analyzer = AttributedAnalyzer(
        llm_client=mock_llm,
        tau_back=0.75,
        use_cache=True
    )

    # Test sample
    test_event = {
        "timestamp": "2025-01-15T10:23:45",
        "event_type": "process_create",
        "source_entity": "cmd.exe",
        "target_entity": "powershell.exe",
        "attributes": {
            "command_line": "powershell.exe -enc SQBFAFgAKABOAGU..."
        }
    }

    xgboost_score = 0.92

    print("\n[TEST] Analyzing sample...")
    result = analyzer.analyze(test_event, xgboost_score, sample_id="test_001")

    print(f"\n[RESULT]")
    print(f"  Decision: {result.decision.value}")
    print(f"  Final Confidence: {result.final_confidence:.4f}")
    print(f"  Latency: {result.latency_ms:.2f} ms")

    print("\n[REPORT]")
    print(json.dumps(result.report, indent=2))

    print("\n[AUDITABLE TRACE]")
    print(result.to_auditable_trace())

    # Test statistics
    stats = analyzer.get_statistics()
    print(f"\n[STATISTICS]")
    print(f"  Total processed: {stats['total_processed']}")
    print(f"  Accepted: {stats['accepted']}")
    print(f"  Manual review: {stats['manual_review']}")
    print(f"  Avg confidence: {stats['average_confidence']:.4f}")

    analyzer.close()
    print("\n[INFO] Test completed")


def demo_manual_review_workflow():
    """Demonstrate the manual review workflow"""
    print("=" * 60)
    print("Demo: Manual Review Workflow")
    print("=" * 60)

    from llm_client import LLMClient, ModelProvider

    mock_llm = LLMClient(provider=ModelProvider.MOCK)
    analyzer = AttributedAnalyzer(llm_client=mock_llm, tau_back=0.85)  # Higher threshold = more reviews

    # Create samples with varying confidence
    samples = [
        ({"event": "suspicious_process_1"}, 0.95),  # High confidence
        ({"event": "suspicious_process_2"}, 0.70),  # Low confidence -> manual review
        ({"event": "suspicious_process_3"}, 0.60),  # Very low confidence -> manual review
    ]

    results = analyzer.analyze_batch(samples)

    print("\n[SAMPLE RESULTS]")
    for r in results:
        print(f"  {r.sample_id}: confidence={r.final_confidence:.3f}, decision={r.decision.value}")

    # Manual review handler
    handler = ManualReviewHandler(analyzer)
    queue_size = handler.refresh_queue()
    print(f"\n[REVIEW QUEUE] {queue_size} samples awaiting manual review")

    # Process reviews
    while True:
        sample = handler.get_next_review()
        if sample is None:
            break

        print(f"\n[REVIEWING] Sample {sample.sample_id}")
        print(f"  Confidence: {sample.final_confidence:.3f}")
        print(f"  Report summary: {sample.report.get('summary', 'N/A')}")

        # Simulate human decision (in practice, this would be a human analyst)
        # For demo, accept if confidence > 0.5, else reject
        human_decision = AttributionDecision.ACCEPT if sample.final_confidence > 0.5 else AttributionDecision.REJECT
        notes = f"Human review: {'Accept' if human_decision == AttributionDecision.ACCEPT else 'Reject'} based on evidence."

        handler.submit_review(sample.sample_id, human_decision, notes)
        print(f"  Human decision: {human_decision.value}")

    # Review statistics
    review_stats = handler.get_review_statistics()
    print(f"\n[REVIEW STATISTICS]")
    print(f"  Total reviews: {review_stats['total_reviews']}")
    print(f"  Accept rate: {review_stats['accept_rate']:.1%}")

    analyzer.close()
    print("\n[INFO] Demo completed")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CAAAPT Attributed Analyzer")
    parser.add_argument("--test", action="store_true", help="Run analyzer tests")
    parser.add_argument("--demo-manual-review", action="store_true", help="Demo manual review workflow")
    parser.add_argument("--tau-back", type=float, default=0.75, help="Confidence threshold")

    args = parser.parse_args()

    if args.demo_manual_review:
        demo_manual_review_workflow()
    elif args.test:
        test_analyzer()
    else:
        print("CAAAPT Attributed Analyzer")
        print("Run with --test to test, --demo-manual-review for manual review demo")