"""
CAAAPT LLM Client
DeepSeek-V2 API wrapper for Chain-of-Thought reasoning and attack attribution
Implements LLM-based backend for four-stage CoT processing as described in Section 3.2

Sanitized version for submission - No hardcoded API keys, internal IPs, or sensitive endpoints
"""

import os
import json
import time
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

# Optional imports - will be available in production environment
try:
    import requests

    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

try:
    from openai import OpenAI

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class ModelProvider(Enum):
    """Supported LLM providers"""
    DEEPSEEK_V2 = "deepseek_v2"
    DEEPSEEK_CHAT = "deepseek_chat"
    OPENAI_COMPATIBLE = "openai_compatible"
    MOCK = "mock"


@dataclass
class LLMRequest:
    """LLM request structure"""
    prompt: str
    system_prompt: Optional[str] = None
    temperature: float = 0.1
    max_tokens: int = 4096
    top_p: float = 0.9
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0


@dataclass
class LLMResponse:
    """LLM response structure"""
    content: str
    model: str
    usage: Dict[str, int]  # prompt_tokens, completion_tokens, total_tokens
    latency_ms: float
    finish_reason: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class LLMClient:
    """
    LLM Client for DeepSeek-V2 API calls

    DeepSeek-V2 pricing (reference):
    - Input: $0.02 per million tokens
    - Output: $0.04 per million tokens

    Per anomalous sample (approx):
    - Prompt (including CoT + retrieved context): ~1,180 tokens
    - Output: ~320 tokens
    - Per invocation cost: ~$9.0e-5
    """

    # Default endpoints (will be configured via environment variables)
    DEFAULT_DEEPSEEK_ENDPOINT = "https://api.deepseek.com/v1"
    DEFAULT_MODEL = "deepseek-chat"

    def __init__(self,
                 api_key: Optional[str] = None,
                 endpoint: Optional[str] = None,
                 model: str = "deepseek-chat",
                 provider: ModelProvider = ModelProvider.DEEPSEEK_V2,
                 timeout_seconds: int = 60,
                 max_retries: int = 3,
                 retry_delay_seconds: float = 1.0):
        """
        Initialize LLM client

        Args:
            api_key: API key for the LLM service (read from env if None)
            endpoint: API endpoint URL (use default if None)
            model: Model name to use
            provider: LLM provider type
            timeout_seconds: Request timeout
            max_retries: Maximum number of retry attempts
            retry_delay_seconds: Delay between retries
        """
        self.model = model
        self.provider = provider
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.retry_delay_seconds = retry_delay_seconds

        # Get API key from environment if not provided
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("LLM_API_KEY")

        # Set endpoint
        if endpoint:
            self.endpoint = endpoint
        elif provider == ModelProvider.DEEPSEEK_V2 or provider == ModelProvider.DEEPSEEK_CHAT:
            self.endpoint = os.environ.get("DEEPSEEK_ENDPOINT", self.DEFAULT_DEEPSEEK_ENDPOINT)
        elif provider == ModelProvider.OPENAI_COMPATIBLE:
            self.endpoint = os.environ.get("OPENAI_ENDPOINT", "https://api.openai.com/v1")
        else:
            self.endpoint = ""

        # Initialize client based on provider
        self._client = None
        self._init_client()

        # Statistics
        self.total_requests = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_latency_ms = 0.0

    def _init_client(self):
        """Initialize the appropriate API client"""
        if self.provider == ModelProvider.MOCK:
            return

        if OPENAI_AVAILABLE and self.provider in [ModelProvider.DEEPSEEK_V2,
                                                  ModelProvider.DEEPSEEK_CHAT,
                                                  ModelProvider.OPENAI_COMPATIBLE]:
            # Use OpenAI-compatible client
            base_url = self.endpoint.rstrip('/') + "/" if not self.endpoint.endswith('/') else self.endpoint
            self._client = OpenAI(
                api_key=self.api_key,
                base_url=base_url,
                timeout=self.timeout_seconds
            )
        elif REQUESTS_AVAILABLE:
            # Use requests as fallback
            self._client = None
        else:
            # No HTTP libraries available - mock mode
            self.provider = ModelProvider.MOCK
            print("[WARN] No HTTP libraries available, using mock provider")

    def _call_openai_compatible(self, request: LLMRequest) -> LLMResponse:
        """Call OpenAI-compatible API"""
        messages = []

        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.append({"role": "user", "content": request.prompt})

        start_time = time.time()

        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            top_p=request.top_p,
            frequency_penalty=request.frequency_penalty,
            presence_penalty=request.presence_penalty
        )

        latency_ms = (time.time() - start_time) * 1000

        # Extract response
        content = response.choices[0].message.content
        usage = {
            "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
            "completion_tokens": response.usage.completion_tokens if response.usage else 0,
            "total_tokens": response.usage.total_tokens if response.usage else 0
        }
        finish_reason = response.choices[0].finish_reason

        return LLMResponse(
            content=content,
            model=response.model,
            usage=usage,
            latency_ms=latency_ms,
            finish_reason=finish_reason
        )

    def _call_requests(self, request: LLMRequest) -> LLMResponse:
        """Call API using requests library"""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        messages = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.append({"role": "user", "content": request.prompt})

        body = {
            "model": self.model,
            "messages": messages,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "top_p": request.top_p,
            "frequency_penalty": request.frequency_penalty,
            "presence_penalty": request.presence_penalty
        }

        start_time = time.time()

        response = requests.post(
            f"{self.endpoint}/chat/completions",
            headers=headers,
            json=body,
            timeout=self.timeout_seconds
        )

        latency_ms = (time.time() - start_time) * 1000
        response.raise_for_status()

        data = response.json()

        content = data["choices"][0]["message"]["content"]
        usage = {
            "prompt_tokens": data.get("usage", {}).get("prompt_tokens", 0),
            "completion_tokens": data.get("usage", {}).get("completion_tokens", 0),
            "total_tokens": data.get("usage", {}).get("total_tokens", 0)
        }
        finish_reason = data["choices"][0].get("finish_reason", "stop")

        return LLMResponse(
            content=content,
            model=data.get("model", self.model),
            usage=usage,
            latency_ms=latency_ms,
            finish_reason=finish_reason
        )

    def _call_mock(self, request: LLMRequest) -> LLMResponse:
        """Generate mock response for testing"""
        latency_ms = 100.0  # Simulate 100ms latency

        # Generate deterministic mock response based on prompt content
        prompt_hash = hashlib.md5(request.prompt.encode()).hexdigest()[:8]

        # Mock token counts (approximate)
        prompt_tokens = len(request.prompt) // 4  # Rough estimate: 4 chars per token
        completion_tokens = 500

        # Generate different mock responses based on prompt content
        if "Stage 1" in request.prompt or "Event Semantic Reconstruction" in request.prompt:
            content = json.dumps({
                "summary": "Suspicious event sequence detected involving process creation and file operations",
                "causal_relationships": [
                    {"cause_event": "process_create", "effect_event": "file_write", "relationship_type": "temporal"}
                ],
                "behavior_sequence": [
                    {"step": 1, "action": "process_create", "target": "suspicious.exe", "source_events": ["event_1"]},
                    {"step": 2, "action": "file_write", "target": "malware.dll", "source_events": ["event_2"]}
                ],
                "anomaly_flags": ["Suspicious parent-child relationship", "Unusual file extension"],
                "causal_graph": {
                    "nodes": ["process_create", "file_write"],
                    "edges": [["process_create", "file_write"]]
                },
                "reconstruction_confidence": 0.85,
                "uncertainties": ["Missing network connection data"]
            }, indent=2)
        elif "Stage 2" in request.prompt or "Tactical Ontology Alignment" in request.prompt:
            content = json.dumps({
                "technique_mappings": [
                    {"behavior_step": 1, "technique_id": "T1059", "technique_name": "Command and Scripting Interpreter",
                     "tactic": "Execution", "evidence": ["Process execution pattern"], "confidence": 0.9}
                ],
                "tactic_sequence": [
                    {"tactic": "Execution", "techniques": ["T1059"], "position": 1, "confidence": 0.85}
                ],
                "alternative_mappings": [],
                "alignment_confidence": 0.82,
                "knowledge_gaps": []
            }, indent=2)
        elif "Stage 3" in request.prompt or "Attack Intention Inference" in request.prompt:
            content = json.dumps({
                "attack_narrative": "Attacker executed malware for initial compromise and persistence establishment",
                "attack_phases": [
                    {"phase": "Initial Compromise", "tactics_involved": ["Execution"],
                     "description": "Malware execution phase"}
                ],
                "suspected_objectives": [
                    {"objective": "Persistence", "confidence": 0.70, "evidence": ["File write to startup directory"]}
                ],
                "campaign_attribution": {
                    "suspected_group": None,
                    "confidence": 0.0,
                    "matching_patterns": []
                },
                "estimated_timeline": {
                    "earliest_action": "recent",
                    "latest_action": "recent",
                    "dwell_time_estimate": "minutes"
                },
                "intention_confidence": 0.75,
                "alternative_hypotheses": [
                    {"hypothesis": "Data staging", "likelihood": "low", "basis": "No exfiltration detected"}
                ]
            }, indent=2)
        else:
            content = json.dumps({
                "component_confidences": {"C_d": 0.8, "C_r": 0.85, "C_a": 0.82, "C_k": 0.75},
                "uncertainty_sources": [],
                "contradictions": [],
                "final_confidence": 0.805,
                "decision": "accept",
                "tau_back_used": 0.75,
                "recommendation": "Attribution accepted with moderate confidence"
            }, indent=2)

        return LLMResponse(
            content=content,
            model="mock-model",
            usage={"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                   "total_tokens": prompt_tokens + completion_tokens},
            latency_ms=latency_ms,
            finish_reason="stop"
        )

    def generate(self, request: LLMRequest) -> LLMResponse:
        """
        Generate LLM response with retry logic

        Args:
            request: LLMRequest object

        Returns:
            LLMResponse object
        """
        for attempt in range(self.max_retries):
            try:
                if self.provider == ModelProvider.MOCK:
                    response = self._call_mock(request)
                elif self.provider in [ModelProvider.DEEPSEEK_V2, ModelProvider.DEEPSEEK_CHAT,
                                       ModelProvider.OPENAI_COMPATIBLE]:
                    if OPENAI_AVAILABLE and self._client:
                        response = self._call_openai_compatible(request)
                    elif REQUESTS_AVAILABLE:
                        response = self._call_requests(request)
                    else:
                        response = self._call_mock(request)
                else:
                    response = self._call_mock(request)

                # Update statistics
                self.total_requests += 1
                self.total_prompt_tokens += response.usage["prompt_tokens"]
                self.total_completion_tokens += response.usage["completion_tokens"]
                self.total_latency_ms += response.latency_ms

                return response

            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise RuntimeError(f"LLM API call failed after {self.max_retries} attempts: {e}")

                wait_time = self.retry_delay_seconds * (2 ** attempt)  # Exponential backoff
                time.sleep(wait_time)

        raise RuntimeError("Unexpected error in retry loop")

    def generate_content(self,
                         prompt: str,
                         system_prompt: Optional[str] = None,
                         temperature: float = 0.1,
                         max_tokens: int = 4096) -> str:
        """
        Simple interface for generating content

        Args:
            prompt: User prompt
            system_prompt: System prompt (optional)
            temperature: Sampling temperature
            max_tokens: Maximum output tokens

        Returns:
            Generated content as string
        """
        request = LLMRequest(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens
        )

        response = self.generate(request)
        return response.content

    def batch_generate(self, prompts: List[str],
                       system_prompt: Optional[str] = None,
                       temperature: float = 0.1,
                       max_tokens: int = 4096) -> List[str]:
        """
        Generate responses for multiple prompts (sequential, not parallel)

        Args:
            prompts: List of user prompts
            system_prompt: System prompt (optional)
            temperature: Sampling temperature
            max_tokens: Maximum output tokens

        Returns:
            List of generated contents
        """
        results = []
        for prompt in prompts:
            content = self.generate_content(prompt, system_prompt, temperature, max_tokens)
            results.append(content)
        return results

    def get_cost_estimate(self) -> Dict[str, float]:
        """
        Estimate cost based on usage statistics

        DeepSeek-V2 pricing:
        - Input: $0.02 per million tokens
        - Output: $0.04 per million tokens

        Returns:
            Dictionary with cost estimates in USD
        """
        input_cost = self.total_prompt_tokens * (0.02 / 1_000_000)
        output_cost = self.total_completion_tokens * (0.04 / 1_000_000)

        return {
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_prompt_tokens + self.total_completion_tokens,
            "input_cost_usd": input_cost,
            "output_cost_usd": output_cost,
            "total_cost_usd": input_cost + output_cost,
            "total_requests": self.total_requests,
            "average_latency_ms": self.total_latency_ms / self.total_requests if self.total_requests > 0 else 0
        }

    def reset_stats(self):
        """Reset usage statistics"""
        self.total_requests = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_latency_ms = 0.0


class APTAttributionLLM:
    """
    Specialized LLM wrapper for APT attribution tasks
    Implements the backend LLM analysis as described in Section 3.2
    """

    def __init__(self, llm_client: LLMClient):
        """
        Initialize APT attribution LLM

        Args:
            llm_client: Configured LLMClient instance
        """
        self.llm_client = llm_client

        # Default system prompts for each stage
        self.system_prompts = {
            "reconstruction": "You are an expert cybersecurity analyst. Analyze system events and reconstruct attack behavior sequences.",
            "alignment": "You are a MITRE ATT&CK framework expert. Map attack behaviors to tactics and techniques.",
            "intention": "You are an APT threat analyst. Infer attacker intentions and attack narratives.",
            "confidence": "You are a security QA analyst. Evaluate confidence of attribution results."
        }

    def call_stage1_reconstruction(self, prompt: str) -> str:
        """Call LLM for Stage 1: Event Semantic Reconstruction"""
        return self.llm_client.generate_content(
            prompt=prompt,
            system_prompt=self.system_prompts["reconstruction"],
            temperature=0.1,
            max_tokens=4096
        )

    def call_stage2_alignment(self, prompt: str) -> str:
        """Call LLM for Stage 2: Tactical Ontology Alignment"""
        return self.llm_client.generate_content(
            prompt=prompt,
            system_prompt=self.system_prompts["alignment"],
            temperature=0.1,
            max_tokens=4096
        )

    def call_stage3_intention(self, prompt: str) -> str:
        """Call LLM for Stage 3: Attack Intention Inference"""
        return self.llm_client.generate_content(
            prompt=prompt,
            system_prompt=self.system_prompts["intention"],
            temperature=0.15,  # Slightly higher for creative inference
            max_tokens=4096
        )

    def call_stage4_confidence(self, prompt: str) -> str:
        """Call LLM for Stage 4: Comprehensive Confidence Evaluation"""
        return self.llm_client.generate_content(
            prompt=prompt,
            system_prompt=self.system_prompts["confidence"],
            temperature=0.05,  # Very low for deterministic scoring
            max_tokens=2048
        )

    def get_cost_statistics(self) -> Dict[str, float]:
        """Get cost statistics from the underlying client"""
        return self.llm_client.get_cost_estimate()


def create_llm_client_from_env() -> LLMClient:
    """
    Create LLM client from environment variables

    Environment variables:
    - LLM_API_KEY or DEEPSEEK_API_KEY: API key
    - LLM_ENDPOINT or DEEPSEEK_ENDPOINT: API endpoint
    - LLM_MODEL: Model name (default: deepseek-chat)
    - LLM_PROVIDER: Provider type (deepseek_v2, deepseek_chat, openai_compatible, mock)
    """
    api_key = os.environ.get("LLM_API_KEY") or os.environ.get("DEEPSEEK_API_KEY")
    endpoint = os.environ.get("LLM_ENDPOINT") or os.environ.get("DEEPSEEK_ENDPOINT")
    model = os.environ.get("LLM_MODEL", "deepseek-chat")

    provider_str = os.environ.get("LLM_PROVIDER", "deepseek_v2").lower()
    provider_map = {
        "deepseek_v2": ModelProvider.DEEPSEEK_V2,
        "deepseek_chat": ModelProvider.DEEPSEEK_CHAT,
        "deepseek": ModelProvider.DEEPSEEK_CHAT,
        "openai_compatible": ModelProvider.OPENAI_COMPATIBLE,
        "mock": ModelProvider.MOCK
    }
    provider = provider_map.get(provider_str, ModelProvider.DEEPSEEK_V2)

    # If no API key and not mock, warn and fallback to mock
    if not api_key and provider != ModelProvider.MOCK:
        print("[WARN] No API key found. Set LLM_API_KEY or DEEPSEEK_API_KEY environment variable.")
        print("[INFO] Falling back to mock provider for testing.")
        provider = ModelProvider.MOCK

    return LLMClient(
        api_key=api_key,
        endpoint=endpoint,
        model=model,
        provider=provider
    )


# ================================================================
# Example Usage and Testing
# ================================================================

def test_llm_client():
    """Test the LLM client functionality"""
    print("=" * 60)
    print("Testing CAAAPT LLM Client")
    print("=" * 60)

    # Create client in mock mode for testing
    client = LLMClient(provider=ModelProvider.MOCK)

    # Test basic generation
    request = LLMRequest(
        prompt="Analyze this suspicious process: powershell.exe -enc base64...",
        system_prompt="You are a cybersecurity analyst.",
        temperature=0.1
    )

    print("\n[TEST] Basic generation:")
    response = client.generate(request)
    print(f"  Content: {response.content[:200]}...")
    print(f"  Latency: {response.latency_ms:.2f} ms")
    print(f"  Tokens: {response.usage}")

    # Test cost estimation
    print("\n[TEST] Cost estimation:")
    cost = client.get_cost_estimate()
    print(f"  Total requests: {cost['total_requests']}")
    print(f"  Total cost: ${cost['total_cost_usd']:.6f}")

    # Test specialized APT attribution LLM
    print("\n[TEST] APT Attribution LLM:")
    apt_llm = APTAttributionLLM(client)

    stage1_result = apt_llm.call_stage1_reconstruction(
        "Event sequence: [2025-01-15 10:23:45] Process Create: powershell.exe (PID 1234)"
    )
    print(f"  Stage 1 result length: {len(stage1_result)} chars")

    print("\n[INFO] LLM client test completed")


def estimate_daily_cost():
    """Estimate daily operational cost as described in Table 8"""
    # Based on THEIA dataset scale with typical tau_front configuration
    # Per anomalous sample: ~1,180 input tokens, ~320 output tokens
    input_tokens_per_sample = 1180
    output_tokens_per_sample = 320

    # DeepSeek-V2 pricing
    input_cost_per_million = 0.02
    output_cost_per_million = 0.04

    # Estimated daily anomalies (based on THEIA dataset scale)
    daily_anomalies = 1250  # Approximate for typical deployment

    daily_input_tokens = daily_anomalies * input_tokens_per_sample
    daily_output_tokens = daily_anomalies * output_tokens_per_sample

    daily_input_cost = daily_input_tokens * (input_cost_per_million / 1_000_000)
    daily_output_cost = daily_output_tokens * (output_cost_per_million / 1_000_000)
    total_daily_cost = daily_input_cost + daily_output_cost

    print("=" * 60)
    print("CAAAPT Daily Cost Estimate (DeepSeek-V2)")
    print("=" * 60)
    print(f"Daily anomalies processed: {daily_anomalies}")
    print(f"Input tokens per sample: {input_tokens_per_sample}")
    print(f"Output tokens per sample: {output_tokens_per_sample}")
    print(f"Daily input tokens: {daily_input_tokens:,}")
    print(f"Daily output tokens: {daily_output_tokens:,}")
    print(f"Input cost: ${daily_input_cost:.4f}")
    print(f"Output cost: ${daily_output_cost:.4f}")
    print(f"Total daily cost: ${total_daily_cost:.4f}")
    print(f"Per thousand anomalies: ${total_daily_cost / (daily_anomalies / 1000):.4f}")
    print(
        f"Per invocation: ${(input_tokens_per_sample * input_cost_per_million + output_tokens_per_sample * output_cost_per_million) / 1_000_000:.6f}")

    return total_daily_cost


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CAAAPT LLM Client for DeepSeek-V2")
    parser.add_argument("--test", action="store_true", help="Run client tests")
    parser.add_argument("--estimate-cost", action="store_true", help="Estimate daily operational cost")
    parser.add_argument("--mock", action="store_true", help="Use mock provider (no API key needed)")

    args = parser.parse_args()

    if args.estimate_cost:
        estimate_daily_cost()
    elif args.test:
        test_llm_client()
    else:
        # Interactive mode
        if args.mock:
            client = LLMClient(provider=ModelProvider.MOCK)
            print("CAAAPT LLM Client (Mock Mode)")
        else:
            client = create_llm_client_from_env()
            print("CAAAPT LLM Client (API Mode)")

        print("Type 'quit' to exit, 'stats' to show cost statistics")

        while True:
            user_input = input("\nEnter prompt: ")
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            if user_input.lower() == 'stats':
                stats = client.get_cost_estimate()
                print(f"Cost stats: {json.dumps(stats, indent=2)}")
                continue

            response = client.generate_content(user_input)
            print(f"\nResponse:\n{response[:500]}...")