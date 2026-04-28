"""
CAAAPT Confidence Calculator
Comprehensive confidence evaluation for APT attribution

Implements the confidence scoring mechanism described in Section 3.2.3:
C_final = a_d * C_d + a_r * C_r + a_a * C_a + a_k * C_k

Where:
- C_d: XGBoost anomaly detection confidence (from frontend screener)
- C_r: Event reconstruction probability (from Stage 1 CoT)
- C_a: Top-k tactic alignment probability (from Stage 2 CoT)
- C_k: RAG knowledge consistency score (retrieved vs. inferred)

Sanitized version for submission - No sensitive paths, API keys, or internal IPs
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum


class ConfidenceLevel(Enum):
    """Confidence level categories"""
    VERY_LOW = "very_low"  # < 0.3
    LOW = "low"  # 0.3 - 0.5
    MEDIUM = "medium"  # 0.5 - 0.7
    HIGH = "high"  # 0.7 - 0.85
    VERY_HIGH = "very_high"  # 0.85 - 0.95
    NEAR_CERTAIN = "near_certain"  # > 0.95


@dataclass
class ComponentConfidence:
    """Individual component confidence scores"""
    C_d: float  # XGBoost anomaly score
    C_r: float  # Reconstruction confidence
    C_a: float  # Alignment confidence
    C_k: float  # Knowledge consistency

    # Weights (sum to 1.0)
    a_d: float = 0.25
    a_r: float = 0.25
    a_a: float = 0.25
    a_k: float = 0.25

    def validate(self) -> bool:
        """Validate confidence values are in [0, 1] range"""
        valid = True
        for v in [self.C_d, self.C_r, self.C_a, self.C_k, self.a_d, self.a_r, self.a_a, self.a_k]:
            if v < 0 or v > 1:
                valid = False
        # Weights should sum to 1.0 (allow small epsilon)
        weight_sum = self.a_d + self.a_r + self.a_a + self.a_k
        if abs(weight_sum - 1.0) > 0.01:
            valid = False
        return valid

    def to_dict(self) -> Dict[str, float]:
        return {
            "C_d": self.C_d,
            "C_r": self.C_r,
            "C_a": self.C_a,
            "C_k": self.C_k,
            "a_d": self.a_d,
            "a_r": self.a_r,
            "a_a": self.a_a,
            "a_k": self.a_k
        }


@dataclass
class FinalConfidence:
    """Final confidence result with detailed breakdown"""
    value: float  # C_final
    level: ConfidenceLevel
    components: ComponentConfidence
    uncertainty_sources: List[Dict[str, Any]]
    contradictions: List[Dict[str, Any]]
    is_reliable: bool
    recommendation: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "value": self.value,
            "level": self.level.value,
            "components": self.components.to_dict(),
            "uncertainty_sources": self.uncertainty_sources,
            "contradictions": self.contradictions,
            "is_reliable": self.is_reliable,
            "recommendation": self.recommendation
        }

    def __str__(self) -> str:
        lines = [
            "=" * 50,
            f"Final Confidence: {self.value:.4f} ({self.level.value})",
            f"Reliable: {self.is_reliable}",
            "-" * 30,
            f"  C_d (XGBoost):  {self.components.C_d:.4f} (weight={self.components.a_d})",
            f"  C_r (Reconstruction): {self.components.C_r:.4f} (weight={self.components.a_r})",
            f"  C_a (Alignment): {self.components.C_a:.4f} (weight={self.components.a_a})",
            f"  C_k (Knowledge): {self.components.C_k:.4f} (weight={self.components.a_k})",
            "-" * 30,
            f"Recommendation: {self.recommendation}",
            "=" * 50
        ]
        return "\n".join(lines)


class ConfidenceCalculator:
    """
    Comprehensive confidence calculator for APT attribution

    Implements Equation (7) from the paper:
    C_final = Phi(C_d, C_r, C_a, C_k; theta)

    Where Phi is a weighted linear combination with optional non-linear adjustments
    """

    def __init__(self,
                 a_d: float = 0.25,
                 a_r: float = 0.25,
                 a_a: float = 0.25,
                 a_k: float = 0.25,
                 tau_reliable: float = 0.75,
                 enable_nonlinear: bool = False):
        """
        Initialize confidence calculator

        Args:
            a_d: Weight for XGBoost anomaly score (C_d)
            a_r: Weight for reconstruction confidence (C_r)
            a_a: Weight for alignment confidence (C_a)
            a_k: Weight for knowledge consistency (C_k)
            tau_reliable: Threshold for considering result reliable
            enable_nonlinear: Apply non-linear adjustments to weights
        """
        self.default_weights = {
            "a_d": a_d,
            "a_r": a_r,
            "a_a": a_a,
            "a_k": a_k
        }
        self.tau_reliable = tau_reliable
        self.enable_nonlinear = enable_nonlinear

    def compute_final_confidence(self,
                                 C_d: float,
                                 C_r: float,
                                 C_a: float,
                                 C_k: float,
                                 weights: Optional[Dict[str, float]] = None,
                                 uncertainty_sources: Optional[List[Dict]] = None,
                                 contradictions: Optional[List[Dict]] = None) -> FinalConfidence:
        """
        Compute final confidence C_final = a_d*C_d + a_r*C_r + a_a*C_a + a_k*C_k

        Args:
            C_d: XGBoost anomaly detection confidence [0, 1]
            C_r: Event reconstruction probability [0, 1]
            C_a: Top-k tactic alignment probability [0, 1]
            C_k: RAG knowledge consistency score [0, 1]
            weights: Optional custom weights (uses defaults if None)
            uncertainty_sources: List of identified uncertainty sources
            contradictions: List of contradictions between stages

        Returns:
            FinalConfidence object with detailed breakdown
        """
        # Clamp values to [0, 1]
        C_d = max(0.0, min(1.0, C_d))
        C_r = max(0.0, min(1.0, C_r))
        C_a = max(0.0, min(1.0, C_a))
        C_k = max(0.0, min(1.0, C_k))

        # Get weights
        if weights is None:
            weights = self.default_weights
        else:
            weights = {k: weights.get(k, self.default_weights[k]) for k in self.default_weights}

        a_d = weights["a_d"]
        a_r = weights["a_r"]
        a_a = weights["a_a"]
        a_k = weights["a_k"]

        # Apply non-linear adjustments if enabled
        if self.enable_nonlinear:
            C_d, C_r, C_a, C_k, a_d, a_r, a_a, a_k = self._apply_nonlinear_adjustments(
                C_d, C_r, C_a, C_k, a_d, a_r, a_a, a_k
            )

        # Compute weighted sum
        final_value = a_d * C_d + a_r * C_r + a_a * C_a + a_k * C_k

        # Determine confidence level
        level = self._get_confidence_level(final_value)

        # Determine reliability
        is_reliable = final_value >= self.tau_reliable

        # Generate recommendation
        recommendation = self._generate_recommendation(final_value, level, uncertainty_sources, contradictions)

        # Create component confidence object
        components = ComponentConfidence(
            C_d=C_d, C_r=C_r, C_a=C_a, C_k=C_k,
            a_d=a_d, a_r=a_r, a_a=a_a, a_k=a_k
        )

        return FinalConfidence(
            value=final_value,
            level=level,
            components=components,
            uncertainty_sources=uncertainty_sources or [],
            contradictions=contradictions or [],
            is_reliable=is_reliable,
            recommendation=recommendation
        )

    def _apply_nonlinear_adjustments(self,
                                     C_d: float, C_r: float, C_a: float, C_k: float,
                                     a_d: float, a_r: float, a_a: float, a_k: float
                                     ) -> Tuple[float, float, float, float, float, float, float, float]:
        """
        Apply non-linear adjustments to confidence values and weights

        Non-linear adjustments include:
        - Penalize when one component is very low (bottleneck effect)
        - Boost when components are consistently high
        - Adjust based on component variance
        """
        components = [C_d, C_r, C_a, C_k]
        weights_list = [a_d, a_r, a_a, a_k]

        # Calculate variance among components
        variance = np.var(components)

        # High variance indicates inconsistency -> reduce confidence
        if variance > 0.1:
            penalty = 0.9
            for i in range(len(components)):
                components[i] = components[i] * penalty

        # If any component is very low (< 0.3), apply additional penalty
        min_component = min(components)
        if min_component < 0.3:
            # Bottleneck effect: overall confidence cannot exceed min_component + 0.2
            max_allowed = min_component + 0.2
            final_value = sum(w * c for w, c in zip(weights_list, components))
            if final_value > max_allowed:
                scale = max_allowed / final_value
                for i in range(len(components)):
                    components[i] = components[i] * scale

        # If all components are high (> 0.8), boost confidence
        if all(c > 0.8 for c in components):
            boost = 1.05
            for i in range(len(components)):
                components[i] = min(1.0, components[i] * boost)

        return tuple(components + weights_list)

    def _get_confidence_level(self, confidence: float) -> ConfidenceLevel:
        """Map confidence value to categorical level"""
        if confidence < 0.3:
            return ConfidenceLevel.VERY_LOW
        elif confidence < 0.5:
            return ConfidenceLevel.LOW
        elif confidence < 0.7:
            return ConfidenceLevel.MEDIUM
        elif confidence < 0.85:
            return ConfidenceLevel.HIGH
        elif confidence < 0.95:
            return ConfidenceLevel.VERY_HIGH
        else:
            return ConfidenceLevel.NEAR_CERTAIN

    def _generate_recommendation(self,
                                 confidence: float,
                                 level: ConfidenceLevel,
                                 uncertainty_sources: Optional[List],
                                 contradictions: Optional[List]) -> str:
        """Generate action recommendation based on confidence"""

        if contradictions and len(contradictions) > 0:
            return "MANUAL_REVIEW_REQUIRED: Contradictions detected between analysis stages"

        if level == ConfidenceLevel.NEAR_CERTAIN:
            return "ACCEPT: Attribution highly reliable"
        elif level == ConfidenceLevel.VERY_HIGH:
            return "ACCEPT: Attribution reliable with minor caveats"
        elif level == ConfidenceLevel.HIGH:
            return "ACCEPT_WITH_REVIEW: Attribution likely correct, recommend human verification"
        elif level == ConfidenceLevel.MEDIUM:
            return "MANUAL_REVIEW: Moderate confidence, human verification required"
        elif level == ConfidenceLevel.LOW:
            return "MANUAL_REVIEW: Low confidence, detailed human analysis needed"
        else:
            return "REJECT: Very low confidence, flag for further investigation"

    def from_stage_outputs(self,
                           xgboost_score: float,
                           reconstruction_output: Dict,
                           alignment_output: Dict,
                           knowledge_consistency: float) -> FinalConfidence:
        """
        Compute final confidence from stage outputs

        This method extracts the necessary confidence values from the output
        dictionaries of each CoT stage.

        Args:
            xgboost_score: C_d from frontend screener
            reconstruction_output: Stage 1 output dict (contains reconstruction_confidence)
            alignment_output: Stage 2 output dict (contains alignment_confidence)
            knowledge_consistency: C_k from RAG consistency check

        Returns:
            FinalConfidence object
        """
        # Extract C_r from reconstruction output
        C_r = reconstruction_output.get("reconstruction_confidence", 0.5)

        # Extract C_a from alignment output
        C_a = alignment_output.get("alignment_confidence", 0.5)

        # Extract uncertainty sources
        uncertainty_sources = reconstruction_output.get("uncertainties", [])
        uncertainty_sources.extend(alignment_output.get("knowledge_gaps", []))

        # Detect contradictions
        contradictions = self._detect_contradictions(reconstruction_output, alignment_output)

        return self.compute_final_confidence(
            C_d=xgboost_score,
            C_r=C_r,
            C_a=C_a,
            C_k=knowledge_consistency,
            uncertainty_sources=uncertainty_sources,
            contradictions=contradictions
        )

    def _detect_contradictions(self,
                               reconstruction_output: Dict,
                               alignment_output: Dict) -> List[Dict]:
        """Detect contradictions between reconstruction and alignment"""
        contradictions = []

        # Check if reconstruction confidence is high but alignment confidence is low
        reconstruction_conf = reconstruction_output.get("reconstruction_confidence", 0.5)
        alignment_conf = alignment_output.get("alignment_confidence", 0.5)

        if reconstruction_conf > 0.8 and alignment_conf < 0.4:
            contradictions.append({
                "type": "confidence_mismatch",
                "description": "High reconstruction confidence but low alignment confidence",
                "severity": "high"
            })

        # Check for conflicting tactic assignments
        behavior_sequence = reconstruction_output.get("behavior_sequence", [])
        tactic_sequence = alignment_output.get("tactic_sequence", [])

        if behavior_sequence and not tactic_sequence:
            contradictions.append({
                "type": "missing_alignment",
                "description": "Behavior sequence detected but no tactic alignment found",
                "severity": "medium"
            })

        return contradictions


class KnowledgeConsistencyScorer:
    """
    Knowledge consistency scorer (C_k)

    Evaluates consistency between:
    - Retrieved knowledge from RAG
    - LLM-inferred attribution
    """

    def __init__(self):
        self.semantic_similarity_threshold = 0.7

    def compute_consistency(self,
                            retrieved_knowledge: List[Dict],
                            inferred_attribution: Dict) -> float:
        """
        Compute knowledge consistency score C_k

        Args:
            retrieved_knowledge: List of knowledge entries from RAG
            inferred_attribution: LLM's inferred attribution result

        Returns:
            Consistency score in [0, 1]
        """
        if not retrieved_knowledge:
            return 0.5  # Neutral score when no knowledge available

        scores = []

        # Check tactic consistency
        inferred_tactics = set(self._extract_tactics(inferred_attribution))
        knowledge_tactics = set(self._extract_tactics_from_knowledge(retrieved_knowledge))

        if knowledge_tactics:
            tactic_overlap = len(inferred_tactics & knowledge_tactics)
            tactic_score = tactic_overlap / max(len(inferred_tactics), 1)
            scores.append(tactic_score)

        # Check technique consistency
        inferred_techniques = set(self._extract_techniques(inferred_attribution))
        knowledge_techniques = set(self._extract_techniques_from_knowledge(retrieved_knowledge))

        if knowledge_techniques:
            technique_overlap = len(inferred_techniques & knowledge_techniques)
            technique_score = technique_overlap / max(len(inferred_techniques), 1)
            scores.append(technique_score)

        # Check semantic similarity with top retrieved entry
        if retrieved_knowledge and inferred_attribution:
            top_knowledge = retrieved_knowledge[0]
            semantic_score = self._compute_semantic_similarity(
                inferred_attribution.get("attack_narrative", ""),
                top_knowledge.get("content", "")
            )
            scores.append(semantic_score)

        if not scores:
            return 0.5

        return np.mean(scores)

    def _extract_tactics(self, attribution: Dict) -> List[str]:
        """Extract tactics from attribution result"""
        tactics = []

        if "tactic_sequence" in attribution:
            for t in attribution["tactic_sequence"]:
                if isinstance(t, dict):
                    tactics.append(t.get("tactic", ""))
                elif isinstance(t, str):
                    tactics.append(t)

        if "technique_mappings" in attribution:
            for m in attribution["technique_mappings"]:
                if isinstance(m, dict) and "tactic" in m:
                    tactics.append(m["tactic"])

        return list(set(tactics))

    def _extract_techniques(self, attribution: Dict) -> List[str]:
        """Extract techniques from attribution result"""
        techniques = []

        if "technique_mappings" in attribution:
            for m in attribution["technique_mappings"]:
                if isinstance(m, dict):
                    if "technique_id" in m:
                        techniques.append(m["technique_id"])
                    if "technique_name" in m:
                        techniques.append(m["technique_name"])

        return list(set(techniques))

    def _extract_tactics_from_knowledge(self, knowledge: List[Dict]) -> List[str]:
        """Extract tactics from retrieved knowledge"""
        tactics = []
        for k in knowledge:
            if "tactic" in k and k["tactic"]:
                tactics.append(k["tactic"])
            if "metadata" in k and "tactic" in k["metadata"]:
                tactics.append(k["metadata"]["tactic"])
        return list(set(tactics))

    def _extract_techniques_from_knowledge(self, knowledge: List[Dict]) -> List[str]:
        """Extract techniques from retrieved knowledge"""
        techniques = []
        for k in knowledge:
            if "technique" in k and k["technique"]:
                techniques.append(k["technique"])
            if "technique_id" in k and k["technique_id"]:
                techniques.append(k["technique_id"])
            if "metadata" in k:
                if "technique" in k["metadata"]:
                    techniques.append(k["metadata"]["technique"])
                if "technique_id" in k["metadata"]:
                    techniques.append(k["metadata"]["technique_id"])
        return list(set(techniques))

    def _compute_semantic_similarity(self, text1: str, text2: str) -> float:
        """
        Compute semantic similarity between two texts

        Uses simple overlap-based similarity. In production, this would
        use embeddings-based similarity.
        """
        if not text1 or not text2:
            return 0.5

        # Simple word overlap similarity (for demonstration)
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.5

        overlap = len(words1 & words2)
        union = len(words1 | words2)

        if union == 0:
            return 0.5

        return overlap / union


class ConfidenceAggregator:
    """
    Aggregate confidence scores from multiple sources
    Useful for ensemble or batch processing
    """

    def __init__(self, calculator: ConfidenceCalculator):
        self.calculator = calculator
        self.history: List[FinalConfidence] = []

    def aggregate(self, confidences: List[FinalConfidence]) -> FinalConfidence:
        """
        Aggregate multiple confidence results

        Args:
            confidences: List of confidence results to aggregate

        Returns:
            Aggregated confidence result
        """
        if not confidences:
            raise ValueError("No confidences to aggregate")

        if len(confidences) == 1:
            return confidences[0]

        # Compute average confidence values
        avg_C_d = np.mean([c.components.C_d for c in confidences])
        avg_C_r = np.mean([c.components.C_r for c in confidences])
        avg_C_a = np.mean([c.components.C_a for c in confidences])
        avg_C_k = np.mean([c.components.C_k for c in confidences])

        # Use weights from first confidence (assuming consistent)
        weights = {
            "a_d": confidences[0].components.a_d,
            "a_r": confidences[0].components.a_r,
            "a_a": confidences[0].components.a_a,
            "a_k": confidences[0].components.a_k
        }

        # Aggregate uncertainty sources
        all_uncertainties = []
        for c in confidences:
            all_uncertainties.extend(c.uncertainty_sources)

        # Aggregate contradictions
        all_contradictions = []
        for c in confidences:
            all_contradictions.extend(c.contradictions)

        # Compute aggregated confidence
        result = self.calculator.compute_final_confidence(
            C_d=avg_C_d,
            C_r=avg_C_r,
            C_a=avg_C_a,
            C_k=avg_C_k,
            weights=weights,
            uncertainty_sources=all_uncertainties,
            contradictions=all_contradictions
        )

        self.history.append(result)
        return result

    def get_aggregated_statistics(self) -> Dict[str, Any]:
        """Get statistics on aggregated confidences"""
        if not self.history:
            return {"message": "No aggregation history"}

        values = [c.value for c in self.history]

        return {
            "count": len(self.history),
            "mean": np.mean(values),
            "std": np.std(values),
            "min": np.min(values),
            "max": np.max(values),
            "reliable_count": sum(1 for c in self.history if c.is_reliable)
        }


# ================================================================
# Factory Functions and Testing
# ================================================================

def create_default_calculator() -> ConfidenceCalculator:
    """Create confidence calculator with default weights (a_d=a_r=a_a=a_k=0.25)"""
    return ConfidenceCalculator(a_d=0.25, a_r=0.25, a_a=0.25, a_k=0.25)


def create_high_precision_calculator() -> ConfidenceCalculator:
    """Create calculator with higher weight on alignment (for high-precision scenarios)"""
    return ConfidenceCalculator(a_d=0.15, a_r=0.20, a_a=0.50, a_k=0.15)


def create_high_recall_calculator() -> ConfidenceCalculator:
    """Create calculator with higher weight on XGBoost (for high-recall scenarios)"""
    return ConfidenceCalculator(a_d=0.50, a_r=0.20, a_a=0.15, a_k=0.15)


def test_confidence_calculator():
    """Test the confidence calculator"""
    print("=" * 60)
    print("Testing Confidence Calculator")
    print("=" * 60)

    # Create calculator
    calculator = create_default_calculator()

    # Test case 1: High confidence scenario
    print("\n[TEST 1] High confidence scenario")
    result1 = calculator.compute_final_confidence(
        C_d=0.92,
        C_r=0.85,
        C_a=0.82,
        C_k=0.88
    )
    print(result1)

    # Test case 2: Low confidence scenario
    print("\n[TEST 2] Low confidence scenario")
    result2 = calculator.compute_final_confidence(
        C_d=0.45,
        C_r=0.40,
        C_a=0.35,
        C_k=0.50
    )
    print(result2)

    # Test case 3: With contradictions
    print("\n[TEST 3] With contradictions")
    result3 = calculator.compute_final_confidence(
        C_d=0.85,
        C_r=0.80,
        C_a=0.35,  # Low alignment confidence
        C_k=0.75,
        contradictions=[{
            "type": "confidence_mismatch",
            "description": "High reconstruction but low alignment",
            "severity": "high"
        }]
    )
    print(result3)

    # Test knowledge consistency scorer
    print("\n[TEST 4] Knowledge consistency scorer")
    scorer = KnowledgeConsistencyScorer()

    retrieved = [
        {"tactic": "Execution", "technique": "T1059", "content": "PowerShell execution pattern"},
        {"tactic": "Defense Evasion", "technique": "T1562", "content": "Disable security tools"}
    ]

    inferred = {
        "tactic_sequence": [{"tactic": "Execution"}],
        "technique_mappings": [{"technique_id": "T1059", "tactic": "Execution"}],
        "attack_narrative": "Attacker used PowerShell for execution"
    }

    C_k = scorer.compute_consistency(retrieved, inferred)
    print(f"  Consistency score (C_k): {C_k:.4f}")

    # Test from stage outputs
    print("\n[TEST 5] From stage outputs")
    reconstruction = {
        "reconstruction_confidence": 0.88,
        "uncertainties": ["Missing network logs"],
        "behavior_sequence": [{"step": 1, "action": "process_create"}]
    }
    alignment = {
        "alignment_confidence": 0.79,
        "knowledge_gaps": ["Limited IOC coverage"],
        "tactic_sequence": [{"tactic": "Execution"}]
    }

    result5 = calculator.from_stage_outputs(
        xgboost_score=0.91,
        reconstruction_output=reconstruction,
        alignment_output=alignment,
        knowledge_consistency=0.85
    )
    print(result5)

    # Test different weight configurations
    print("\n[TEST 6] Different weight configurations")
    high_precision_calc = create_high_precision_calculator()
    high_recall_calc = create_high_recall_calculator()

    C_d, C_r, C_a, C_k = 0.80, 0.75, 0.90, 0.70

    default_result = calculator.compute_final_confidence(C_d, C_r, C_a, C_k)
    precision_result = high_precision_calc.compute_final_confidence(C_d, C_r, C_a, C_k)
    recall_result = high_recall_calc.compute_final_confidence(C_d, C_r, C_a, C_k)

    print(f"  Default (0.25 each):    {default_result.value:.4f}")
    print(f"  High Precision (0.5 alignment): {precision_result.value:.4f}")
    print(f"  High Recall (0.5 XGBoost):   {recall_result.value:.4f}")

    print("\n[INFO] Confidence calculator test completed")


def demo_confidence_thresholds():
    """Demonstrate confidence threshold behavior for decision making"""
    print("=" * 60)
    print("Confidence Threshold Demonstration")
    print("=" * 60)

    calculator = create_default_calculator()

    thresholds = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9]

    print("\n[THRESHOLD ANALYSIS]")
    print(f"{'Confidence':<12} {'0.5':<12} {'0.6':<12} {'0.7':<12} {'0.75':<12} {'0.8':<12} {'0.85':<12} {'0.9':<12}")
    print("-" * 96)

    test_cases = [
        (0.95, 0.90, 0.92, 0.93, "Very High"),
        (0.85, 0.80, 0.82, 0.83, "High"),
        (0.70, 0.68, 0.72, 0.70, "Medium"),
        (0.55, 0.50, 0.52, 0.53, "Low"),
        (0.30, 0.35, 0.32, 0.33, "Very Low"),
    ]

    for C_d, C_r, C_a, C_k, label in test_cases:
        result = calculator.compute_final_confidence(C_d, C_r, C_a, C_k)
        decisions = []
        for tau in thresholds:
            if result.value >= tau:
                decisions.append("ACCEPT")
            else:
                decisions.append("REVIEW")

        print(
            f"{label:<12} {decisions[0]:<12} {decisions[1]:<12} {decisions[2]:<12} {decisions[3]:<12} {decisions[4]:<12} {decisions[5]:<12} {decisions[6]:<12}")

    print("\n[SUMMARY]")
    print("For tau_back = 0.75 (default from paper):")
    print("  - Confidence >= 0.75: Accept attribution")
    print("  - Confidence < 0.75: Trigger manual review")
    print("  - Very low confidence (< 0.5): Reject/flag for investigation")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CAAAPT Confidence Calculator")
    parser.add_argument("--test", action="store_true", help="Run confidence calculator tests")
    parser.add_argument("--demo", action="store_true", help="Run threshold demonstration")

    args = parser.parse_args()

    if args.demo:
        demo_confidence_thresholds()
    elif args.test:
        test_confidence_calculator()
    else:
        print("CAAAPT Confidence Calculator")
        print("Run with --test to test, --demo for threshold demonstration")