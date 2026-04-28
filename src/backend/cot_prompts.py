"""
CAAAPT Chain-of-Thought Prompts
Four-stage structured CoT reasoning mechanism for APT attribution
As described in Section 3.2.3 of the paper

Stage 1: Event Semantic Reconstruction
Stage 2: Tactical Ontology Alignment
Stage 3: Attack Intention Inference
Stage 4: Comprehensive Confidence Evaluation

Sanitized version for submission - No sensitive paths, API keys, or internal IPs
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json
import re


@dataclass
class CoTStageOutput:
    """Output structure for each CoT stage"""
    stage_name: str
    output: str
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class CoTPromptTemplates:
    """
    Centralized collection of Chain-of-Thought prompt templates
    Each template follows the structured reasoning approach described in Section 3.2.3
    """

    # ================================================================
    # STAGE 1: Event Semantic Reconstruction
    # ================================================================
    EVENT_RECONSTRUCTION_TEMPLATE = """
You are an expert cybersecurity analyst specializing in APT attack investigation. Your task is to reconstruct the semantic meaning of a sequence of system events.

## Input Context
### Suspicious Event Sequence:
{event_sequence}

### Retrieved Knowledge Base Context:
{knowledge_context}

### ATT&CK Framework Reference:
{tactic_reference}

## Task: Event Semantic Reconstruction
Please analyze the event sequence and reconstruct the high-level behavior. Follow this reasoning process:

1. **Event Summarization**: Briefly describe what happened in this event sequence (2-3 sentences).

2. **Causal Relationship Analysis**: Identify temporal and logical dependencies between events.
   - Which events are causally related?
   - What is the temporal ordering?
   - Are there any missing intermediate steps?

3. **Behavior Abstraction**: Abstract low-level events into high-level behaviors.
   - Group related events into behavior units
   - Identify the action (e.g., "file download", "process injection", "registry modification")
   - Identify the target (e.g., which file, process, or system resource)

4. **Anomaly Identification**: Highlight which behaviors deviate from normal patterns.

5. **Causal Graph Construction**: Build a step-by-step behavior chain.

Output format (JSON):
{
    "summary": "string",
    "causal_relationships": [
        {"cause_event": "event_X", "effect_event": "event_Y", "relationship_type": "temporal|logical|data_flow"}
    ],
    "behavior_sequence": [
        {"step": 1, "action": "string", "target": "string", "source_events": ["event_1", "event_2"]}
    ],
    "anomaly_flags": ["flag1", "flag2"],
    "causal_graph": {
        "nodes": ["behavior_1", "behavior_2"],
        "edges": [["behavior_1", "behavior_2"]]
    },
    "reconstruction_confidence": 0.0-1.0,
    "uncertainties": ["list of aspects that need further verification"]
}

## Important Guidelines:
- If information is insufficient, state uncertainties explicitly
- Do not fabricate events not present in the sequence
- Base causal inference on temporal ordering and data flow patterns
- Confidence should reflect the completeness of evidence

Begin your analysis:
"""

    # ================================================================
    # STAGE 2: Tactical Ontology Alignment
    # ================================================================
    TACTIC_ALIGNMENT_TEMPLATE = """
You are an expert in MITRE ATT&CK framework mapping. Your task is to align the reconstructed attack behaviors with known tactics and techniques.

## Input: Reconstructed Behavior Sequence
{behavior_sequence}

## Input: Causal Graph
{causal_graph}

## Retrieved Knowledge:
{knowledge_context}

## MITRE ATT&CK Tactics Reference:
{tactic_reference}

## Task: Tactical Ontology Alignment
Map the behavior sequence to MITRE ATT&CK tactics using the following reasoning:

1. **Technique Identification**: For each behavior, identify matching ATT&CK techniques.
   - What technique does this behavior correspond to?
   - What is the technique ID (e.g., T1059)?
   - What is the evidence supporting this mapping?

2. **Tactic Classification**: Group techniques under their parent tactics.
   - Initial Access | Execution | Persistence | Privilege Escalation
   - Defense Evasion | Credential Access | Discovery | Lateral Movement
   - Collection | Command and Control | Exfiltration | Impact

3. **Confidence Scoring**: For each tactic assignment, provide confidence based on:
   - Direct indicator match (high)
   - Behavioral pattern match (medium)
   - Temporal/causal context inference (low)

4. **Alternative Mappings**: Consider possible alternative tactic interpretations.

Output format (JSON):
{
    "technique_mappings": [
        {
            "behavior_step": 1,
            "technique_id": "Txxxx",
            "technique_name": "string",
            "tactic": "string",
            "evidence": ["evidence1", "evidence2"],
            "confidence": 0.0-1.0
        }
    ],
    "tactic_sequence": [
        {
            "tactic": "string",
            "techniques": ["Txxxx", "Tyyyy"],
            "position": 1,
            "confidence": 0.0-1.0
        }
    ],
    "alternative_mappings": [
        {
            "behavior_step": 1,
            "alternative_tactic": "string",
            "alternative_technique": "Txxxx",
            "reasoning": "string"
        }
    ],
    "alignment_confidence": 0.0-1.0,
    "knowledge_gaps": ["missing information that would improve alignment"]
}

## Important Guidelines:
- Base mappings on observed behaviors, not assumptions
- If multiple mappings are plausible, list alternatives
- Lower confidence when evidence is indirect
- Use ATT&CK technique IDs precisely

Begin alignment:
"""

    # ================================================================
    # STAGE 3: Attack Intention Inference
    # ================================================================
    INTENT_INFERENCE_TEMPLATE = """
You are an expert in adversarial reasoning and APT attack analysis. Your task is to infer the attacker's high-level goals and attack narrative.

## Input: Identified Tactics and Techniques
{tactic_sequence}

## Input: Behavior Sequence
{behavior_sequence}

## Input: Retrieved Attack Scenarios
{attack_scenarios}

## Task: Attack Intention Inference
Infer the attacker's strategic intent using the following reasoning:

1. **Attack Narrative Construction**: Build a coherent story of the attack.
   - What is the attacker trying to achieve?
   - What is the likely entry point?
   - How does each tactic contribute to the overall goal?

2. **Campaign Attribution** (if possible):
   - Does this pattern match known APT groups?
   - What are the signature behaviors?
   - What is the confidence of attribution?

3. **Attack Phase Identification**:
   - Reconnaissance | Initial Compromise | Establish Foothold
   - Escalate Privileges | Internal Reconnaissance | Lateral Movement
   - Maintain Presence | Complete Mission | Exfiltrate | Cover Tracks

4. **Objective Inference**:
   - Data theft | System destruction | Persistence | Espionage
   - Ransomware | Cryptojacking | Botnet | Other

5. **Temporal Reasoning**: What is the likely timeline of attacker actions?

Output format (JSON):
{
    "attack_narrative": "string describing the attack story in 3-5 sentences",
    "attack_phases": [
        {"phase": "string", "tactics_involved": ["tactic1", "tactic2"], "description": "string"}
    ],
    "suspected_objectives": [
        {"objective": "string", "confidence": 0.0-1.0, "evidence": ["evidence1"]}
    ],
    "campaign_attribution": {
        "suspected_group": "string or null",
        "confidence": 0.0-1.0,
        "matching_patterns": ["pattern1", "pattern2"]
    },
    "estimated_timeline": {
        "earliest_action": "timestamp or relative",
        "latest_action": "timestamp or relative",
        "dwell_time_estimate": "string"
    },
    "intention_confidence": 0.0-1.0,
    "alternative_hypotheses": [
        {"hypothesis": "string", "likelihood": "low|medium|high", "basis": "string"}
    ]
}

## Important Guidelines:
- Distinguish between observed behavior and inferred intent
- Multiple objectives may coexist
- If attribution is uncertain, state "unknown" rather than guessing
- Provide evidence chains for attribution claims

Begin intention inference:
"""

    # ================================================================
    # STAGE 4: Comprehensive Confidence Evaluation
    # ================================================================
    CONFIDENCE_EVALUATION_TEMPLATE = """
You are a quality assurance analyst for security detection systems. Your task is to evaluate the overall confidence of the attribution result using multi-source uncertainty quantification.

## Input: Stage 1 Reconstruction Output
{reconstruction_output}

## Input: Stage 2 Alignment Output
{alignment_output}

## Input: Stage 3 Intention Output
{intention_output}

## Input: XGBoost Anomaly Score
{xgboost_score}

## Input: RAG Knowledge Consistency Metrics
{knowledge_consistency}

## Task: Comprehensive Confidence Evaluation
Compute the final confidence score C_final = Phi(C_d, C_r, C_a, C_k; theta) where:

- C_d: XGBoost anomaly detection confidence (from frontend screener)
- C_r: Event reconstruction probability (from Stage 1)
- C_a: Top-k tactic alignment probability (from Stage 2)
- C_k: RAG knowledge consistency score (from retrieved vs. inferred)

Follow this reasoning:

1. **Component Confidence Extraction**:
   - Extract C_d from the provided XGBoost score
   - Extract C_r from Stage 1 reconstruction_confidence
   - Extract C_a from Stage 2 alignment_confidence
   - Compute C_k based on knowledge consistency

2. **Uncertainty Aggregation**:
   - Identify sources of uncertainty in each stage
   - Check for contradictions between stages
   - Evaluate evidence sufficiency

3. **Final Confidence Calculation**:
   - Weighted combination: C_final = a_d*C_d + a_r*C_r + a_a*C_a + a_k*C_k
   - Default weights: a_d=0.25, a_r=0.25, a_a=0.25, a_k=0.25
   - Adjust weights if specific components are unavailable

4. **Decision Recommendation**:
   - If C_final >= tau_back (default 0.75): Accept attribution
   - If C_final < tau_back: Trigger manual review

Output format (JSON):
{
    "component_confidences": {
        "C_d": 0.0-1.0,
        "C_r": 0.0-1.0,
        "C_a": 0.0-1.0,
        "C_k": 0.0-1.0,
        "weights": {"a_d": 0.25, "a_r": 0.25, "a_a": 0.25, "a_k": 0.25}
    },
    "uncertainty_sources": [
        {"source": "string", "severity": "low|medium|high", "description": "string"}
    ],
    "contradictions": [
        {"between": ["stage1", "stage2"], "description": "string"}
    ],
    "final_confidence": 0.0-1.0,
    "decision": "accept|manual_review|reject",
    "tau_back_used": 0.75,
    "recommendation": "string explaining the decision"
}

## Important Guidelines:
- Be conservative when evidence is contradictory
- Document all uncertainty sources explicitly
- The final decision should align with the confidence threshold

Begin confidence evaluation:
"""

    # ================================================================
    # Helper Methods
    # ================================================================

    @classmethod
    def get_stage1_prompt(cls, event_sequence: str, knowledge_context: str,
                          tactic_reference: str = "") -> str:
        """Generate Stage 1 prompt with filled placeholders"""
        return cls.EVENT_RECONSTRUCTION_TEMPLATE.format(
            event_sequence=event_sequence,
            knowledge_context=knowledge_context,
            tactic_reference=tactic_reference or cls._get_default_tactic_reference()
        )

    @classmethod
    def get_stage2_prompt(cls, behavior_sequence: str, causal_graph: str,
                          knowledge_context: str, tactic_reference: str = "") -> str:
        """Generate Stage 2 prompt with filled placeholders"""
        return cls.TACTIC_ALIGNMENT_TEMPLATE.format(
            behavior_sequence=behavior_sequence,
            causal_graph=causal_graph,
            knowledge_context=knowledge_context,
            tactic_reference=tactic_reference or cls._get_default_tactic_reference()
        )

    @classmethod
    def get_stage3_prompt(cls, tactic_sequence: str, behavior_sequence: str,
                          attack_scenarios: str = "") -> str:
        """Generate Stage 3 prompt with filled placeholders"""
        return cls.INTENT_INFERENCE_TEMPLATE.format(
            tactic_sequence=tactic_sequence,
            behavior_sequence=behavior_sequence,
            attack_scenarios=attack_scenarios or "No specific attack scenarios retrieved."
        )

    @classmethod
    def get_stage4_prompt(cls, reconstruction_output: str, alignment_output: str,
                          intention_output: str, xgboost_score: float,
                          knowledge_consistency: str = "") -> str:
        """Generate Stage 4 prompt with filled placeholders"""
        return cls.CONFIDENCE_EVALUATION_TEMPLATE.format(
            reconstruction_output=reconstruction_output,
            alignment_output=alignment_output,
            intention_output=intention_output,
            xgboost_score=str(xgboost_score),
            knowledge_consistency=knowledge_consistency or "Knowledge consistency: normal"
        )

    @staticmethod
    def _get_default_tactic_reference() -> str:
        """Get default MITRE ATT&CK tactic reference"""
        return """
MITRE ATT&CK Tactics (TA0001-TA0011):
- TA0001: Initial Access - Techniques to get initial foothold
- TA0002: Execution - Techniques to run malicious code
- TA0003: Persistence - Techniques to maintain presence
- TA0004: Privilege Escalation - Techniques to gain higher permissions
- TA0005: Defense Evasion - Techniques to avoid detection
- TA0006: Credential Access - Techniques to steal credentials
- TA0007: Discovery - Techniques to explore environment
- TA0008: Lateral Movement - Techniques to move through systems
- TA0009: Collection - Techniques to gather data of interest
- TA0010: Exfiltration - Techniques to steal data
- TA0011: Command and Control - Techniques to communicate with compromised systems
"""


class ChainOfThoughtReasoner:
    """
    Chain-of-Thought reasoner that orchestrates the four-stage reasoning process
    """

    def __init__(self, llm_client=None):
        """
        Initialize the CoT reasoner

        Args:
            llm_client: LLM client for generating responses (e.g., DeepSeek-V2 API)
                        If None, returns prompts for manual testing
        """
        self.llm_client = llm_client
        self.templates = CoTPromptTemplates()
        self.stage_outputs: List[CoTStageOutput] = []

    def stage1_reconstruct(self, event_sequence: str, knowledge_context: str,
                           tactic_reference: str = "") -> Dict[str, Any]:
        """
        Stage 1: Event Semantic Reconstruction

        Returns parsed JSON output from LLM
        """
        prompt = self.templates.get_stage1_prompt(
            event_sequence=event_sequence,
            knowledge_context=knowledge_context,
            tactic_reference=tactic_reference
        )

        if self.llm_client is None:
            # Return prompt for manual testing
            return {"prompt": prompt, "note": "LLM client not configured"}

        response = self._call_llm(prompt)
        parsed = self._parse_json_response(response)

        self.stage_outputs.append(CoTStageOutput(
            stage_name="stage1_reconstruction",
            output=response,
            confidence=parsed.get("reconstruction_confidence", 0.5),
            metadata=parsed
        ))

        return parsed

    def stage2_align(self, behavior_sequence: str, causal_graph: str,
                     knowledge_context: str, tactic_reference: str = "") -> Dict[str, Any]:
        """
        Stage 2: Tactical Ontology Alignment

        Returns parsed JSON output from LLM
        """
        prompt = self.templates.get_stage2_prompt(
            behavior_sequence=behavior_sequence,
            causal_graph=causal_graph,
            knowledge_context=knowledge_context,
            tactic_reference=tactic_reference
        )

        if self.llm_client is None:
            return {"prompt": prompt, "note": "LLM client not configured"}

        response = self._call_llm(prompt)
        parsed = self._parse_json_response(response)

        self.stage_outputs.append(CoTStageOutput(
            stage_name="stage2_alignment",
            output=response,
            confidence=parsed.get("alignment_confidence", 0.5),
            metadata=parsed
        ))

        return parsed

    def stage3_infer(self, tactic_sequence: str, behavior_sequence: str,
                     attack_scenarios: str = "") -> Dict[str, Any]:
        """
        Stage 3: Attack Intention Inference

        Returns parsed JSON output from LLM
        """
        prompt = self.templates.get_stage3_prompt(
            tactic_sequence=tactic_sequence,
            behavior_sequence=behavior_sequence,
            attack_scenarios=attack_scenarios
        )

        if self.llm_client is None:
            return {"prompt": prompt, "note": "LLM client not configured"}

        response = self._call_llm(prompt)
        parsed = self._parse_json_response(response)

        self.stage_outputs.append(CoTStageOutput(
            stage_name="stage3_intention",
            output=response,
            confidence=parsed.get("intention_confidence", 0.5),
            metadata=parsed
        ))

        return parsed

    def stage4_evaluate(self, reconstruction_output: Dict, alignment_output: Dict,
                        intention_output: Dict, xgboost_score: float,
                        knowledge_consistency: str = "") -> Dict[str, Any]:
        """
        Stage 4: Comprehensive Confidence Evaluation

        Returns parsed JSON output from LLM
        """
        prompt = self.templates.get_stage4_prompt(
            reconstruction_output=json.dumps(reconstruction_output, indent=2),
            alignment_output=json.dumps(alignment_output, indent=2),
            intention_output=json.dumps(intention_output, indent=2),
            xgboost_score=xgboost_score,
            knowledge_consistency=knowledge_consistency
        )

        if self.llm_client is None:
            return {"prompt": prompt, "note": "LLM client not configured"}

        response = self._call_llm(prompt)
        parsed = self._parse_json_response(response)

        self.stage_outputs.append(CoTStageOutput(
            stage_name="stage4_confidence",
            output=response,
            confidence=parsed.get("final_confidence", 0.5),
            metadata=parsed
        ))

        return parsed

    def run_full_pipeline(self, event_sequence: str, knowledge_context: str,
                          xgboost_score: float, tactic_reference: str = "") -> Dict[str, Any]:
        """
        Run all four stages sequentially

        Args:
            event_sequence: Raw event sequence to analyze
            knowledge_context: Retrieved knowledge for RAG
            xgboost_score: Anomaly score from frontend screener
            tactic_reference: Optional tactic reference

        Returns:
            Complete attribution result with all stage outputs
        """
        # Stage 1: Reconstruction
        reconstruction = self.stage1_reconstruct(
            event_sequence=event_sequence,
            knowledge_context=knowledge_context,
            tactic_reference=tactic_reference
        )

        if "prompt" in reconstruction:
            return {"error": "LLM client not configured", "stages": self.stage_outputs}

        # Stage 2: Alignment
        alignment = self.stage2_align(
            behavior_sequence=json.dumps(reconstruction.get("behavior_sequence", []), indent=2),
            causal_graph=json.dumps(reconstruction.get("causal_graph", {}), indent=2),
            knowledge_context=knowledge_context,
            tactic_reference=tactic_reference
        )

        # Stage 3: Intention
        intention = self.stage3_infer(
            tactic_sequence=json.dumps(alignment.get("tactic_sequence", []), indent=2),
            behavior_sequence=json.dumps(reconstruction.get("behavior_sequence", []), indent=2)
        )

        # Stage 4: Confidence
        confidence = self.stage4_evaluate(
            reconstruction_output=reconstruction,
            alignment_output=alignment,
            intention_output=intention,
            xgboost_score=xgboost_score
        )

        return {
            "reconstruction": reconstruction,
            "alignment": alignment,
            "intention": intention,
            "confidence": confidence,
            "final_decision": confidence.get("decision", "unknown")
        }

    def _call_llm(self, prompt: str) -> str:
        """
        Call LLM API (placeholder - implement based on actual LLM service)

        For DeepSeek-V2 or other LLM services, implement the actual API call here.
        """
        if self.llm_client is None:
            raise ValueError("LLM client not configured")

        # Placeholder for actual LLM API call
        # Example implementation for DeepSeek-V2:
        #
        # import requests
        # response = requests.post(
        #     "https://api.deepseek.com/v1/chat/completions",
        #     headers={"Authorization": f"Bearer {self.api_key}"},
        #     json={
        #         "model": "deepseek-chat",
        #         "messages": [{"role": "user", "content": prompt}],
        #         "temperature": 0.1
        #     }
        # )
        # return response.json()["choices"][0]["message"]["content"]

        # For now, return a mock response
        return self._mock_llm_response(prompt)

    def _mock_llm_response(self, prompt: str) -> str:
        """Generate mock LLM response for testing"""
        if "Stage 1" in prompt or "Event Semantic Reconstruction" in prompt:
            return json.dumps({
                "summary": "Suspicious process creation followed by file download and execution",
                "causal_relationships": [
                    {"cause_event": "process_create", "effect_event": "file_download", "relationship_type": "temporal"}
                ],
                "behavior_sequence": [
                    {"step": 1, "action": "process_create", "target": "powershell.exe", "source_events": ["evt1"]},
                    {"step": 2, "action": "file_download", "target": "malware.exe", "source_events": ["evt2"]}
                ],
                "anomaly_flags": ["Unusual parent process", "Suspicious download source"],
                "causal_graph": {"nodes": ["process_create", "file_download"],
                                 "edges": [["process_create", "file_download"]]},
                "reconstruction_confidence": 0.85,
                "uncertainties": ["Download source not fully verified"]
            })
        elif "Stage 2" in prompt or "Tactical Ontology Alignment" in prompt:
            return json.dumps({
                "technique_mappings": [
                    {"behavior_step": 1, "technique_id": "T1059", "technique_name": "Command and Scripting Interpreter",
                     "tactic": "Execution", "evidence": ["PowerShell execution"], "confidence": 0.9}
                ],
                "tactic_sequence": [
                    {"tactic": "Execution", "techniques": ["T1059"], "position": 1, "confidence": 0.85}
                ],
                "alternative_mappings": [],
                "alignment_confidence": 0.82,
                "knowledge_gaps": []
            })
        elif "Stage 3" in prompt or "Attack Intention Inference" in prompt:
            return json.dumps({
                "attack_narrative": "Attacker gained initial access and executed malware for credential theft",
                "attack_phases": [
                    {"phase": "Initial Compromise", "tactics_involved": ["Execution"],
                     "description": "Malware execution"}
                ],
                "suspected_objectives": [
                    {"objective": "Credential Access", "confidence": 0.75, "evidence": ["Suspicious process behavior"]}
                ],
                "campaign_attribution": {"suspected_group": null, "confidence": 0.3, "matching_patterns": []},
                "estimated_timeline": {"earliest_action": "recent", "latest_action": "recent",
                                       "dwell_time_estimate": "minutes"},
                "intention_confidence": 0.78,
                "alternative_hypotheses": []
            })
        else:
            return json.dumps({
                "component_confidences": {"C_d": 0.8, "C_r": 0.85, "C_a": 0.82, "C_k": 0.75,
                                          "weights": {"a_d": 0.25, "a_r": 0.25, "a_a": 0.25, "a_k": 0.25}},
                "uncertainty_sources": [{"source": "Knowledge base coverage", "severity": "low", "description": ""}],
                "contradictions": [],
                "final_confidence": 0.805,
                "decision": "accept",
                "tau_back_used": 0.75,
                "recommendation": "Attribution accepted with moderate confidence"
            })

    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON from LLM response"""
        try:
            # Try to extract JSON from response
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                return json.loads(json_match.group())
            return json.loads(response)
        except json.JSONDecodeError:
            return {"raw_response": response, "parse_error": True}

    def get_full_reasoning_trace(self) -> str:
        """Get the complete reasoning trace from all stages"""
        trace_parts = []
        for output in self.stage_outputs:
            trace_parts.append(f"=== {output.stage_name.upper()} ===\n{output.output}\n")
            trace_parts.append(f"Confidence: {output.confidence}\n")
        return "\n".join(trace_parts)

    def clear_history(self):
        """Clear stage output history"""
        self.stage_outputs = []


# ================================================================
# Utility Functions
# ================================================================

def load_prompt_from_file(file_path: str) -> str:
    """Load prompt template from file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


def save_prompt_to_file(prompt: str, file_path: str):
    """Save prompt template to file"""
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(prompt)


def create_prompt_files(output_dir: str = "prompts"):
    """Create individual prompt template files"""
    import os
    os.makedirs(output_dir, exist_ok=True)

    templates = [
        ("event_reconstruction.txt", CoTPromptTemplates.EVENT_RECONSTRUCTION_TEMPLATE),
        ("tactic_alignment.txt", CoTPromptTemplates.TACTIC_ALIGNMENT_TEMPLATE),
        ("intent_inference.txt", CoTPromptTemplates.INTENT_INFERENCE_TEMPLATE),
        ("confidence_eval.txt", CoTPromptTemplates.CONFIDENCE_EVALUATION_TEMPLATE),
    ]

    for filename, content in templates:
        filepath = os.path.join(output_dir, filename)
        save_prompt_to_file(content, filepath)
        print(f"Created: {filepath}")


# ================================================================
# Main Entry Point
# ================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CAAAPT Chain-of-Thought Prompts")
    parser.add_argument("--create-files", action="store_true",
                        help="Create individual prompt template files")
    parser.add_argument("--output-dir", type=str, default="prompts",
                        help="Output directory for prompt files")
    parser.add_argument("--test", action="store_true",
                        help="Run test with mock LLM")

    args = parser.parse_args()

    if args.create_files:
        create_prompt_files(args.output_dir)
        print(f"Prompt files created in {args.output_dir}")

    if args.test:
        print("=" * 60)
        print("Testing Chain-of-Thought Reasoner")
        print("=" * 60)

        reasoner = ChainOfThoughtReasoner(llm_client="mock")

        result = reasoner.run_full_pipeline(
            event_sequence="[2025-01-15 10:23:45] Process Create: powershell.exe (PID 1234) from parent cmd.exe",
            knowledge_context="Retrieved: PowerShell is commonly used for fileless malware execution",
            xgboost_score=0.92
        )

        print("\n[RESULT] Full pipeline output:")
        print(json.dumps(result, indent=2))

        print("\n[REASONING TRACE]")
        print(reasoner.get_full_reasoning_trace())