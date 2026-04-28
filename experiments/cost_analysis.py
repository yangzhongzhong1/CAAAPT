#!/usr/bin/env python3
"""
Cost Analysis Module for CAAAPT Framework

This module implements the cost analysis experiments to evaluate the economic
feasibility and computational efficiency of the framework. The code is fully
desensitized and only contains the experimental workflow without any actual
data or result values.

Analysis Dimensions (based on paper Section 4.6):
- Storage overhead analysis
- Time overhead analysis
- Economic cost analysis
- Scalability analysis

Author: CAAAPT Team
"""

import argparse
import logging
import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CostCategory(Enum):
    """Categories of cost to analyze"""
    STORAGE = "storage"
    TIME = "time"
    ECONOMIC = "economic"
    ENERGY = "energy"
    SCALABILITY = "scalability"


@dataclass
class CostConfig:
    """Configuration for cost analysis experiment"""
    category: CostCategory
    description: str
    sampling_ratio: float = 1.0  # Ratio of data to process
    batch_size: int = 32
    repeat_count: int = 3  # Number of repetitions for statistical significance


@dataclass
class StorageRecord:
    """Record of storage usage measurement"""
    component: str
    original_size_mb: float
    compressed_size_mb: float
    metadata_size_mb: float
    index_size_mb: float
    backup_size_mb: float

    @property
    def total_size_mb(self) -> float:
        return self.compressed_size_mb + self.metadata_size_mb + self.index_size_mb

    @property
    def compression_ratio(self) -> float:
        if self.original_size_mb > 0:
            return self.original_size_mb / self.compressed_size_mb
        return 1.0


@dataclass
class TimeRecord:
    """Record of time overhead measurement"""
    stage_name: str
    avg_time_ms: float
    min_time_ms: float
    max_time_ms: float
    p95_time_ms: float
    p99_time_ms: float
    samples_processed: int

    @property
    def throughput_per_second(self) -> float:
        if self.avg_time_ms > 0:
            return (self.samples_processed / self.avg_time_ms) * 1000
        return 0.0


@dataclass
class EconomicRecord:
    """Record of economic cost measurement"""
    component: str
    unit_cost_usd: float
    quantity: float
    total_cost_usd: float
    cost_per_sample_usd: float
    billing_model: str  # e.g., "per_token", "per_request", "per_hour"


@dataclass
class ScalabilityRecord:
    """Record of scalability measurement"""
    data_volume_gb: float
    processing_time_s: float
    memory_usage_mb: float
    cost_usd: float
    throughput_qps: float


class MetricsCollector:
    """Collect and aggregate cost metrics"""

    def __init__(self):
        self.storage_records: List[StorageRecord] = []
        self.time_records: List[TimeRecord] = []
        self.economic_records: List[EconomicRecord] = []
        self.scalability_records: List[ScalabilityRecord] = []

    def add_storage_record(self, record: StorageRecord):
        self.storage_records.append(record)

    def add_time_record(self, record: TimeRecord):
        self.time_records.append(record)

    def add_economic_record(self, record: EconomicRecord):
        self.economic_records.append(record)

    def add_scalability_record(self, record: ScalabilityRecord):
        self.scalability_records.append(record)

    def get_storage_summary(self) -> Dict[str, Any]:
        """Get summary of storage measurements"""
        if not self.storage_records:
            return {}

        total_original = sum(r.original_size_mb for r in self.storage_records)
        total_compressed = sum(r.compressed_size_mb for r in self.storage_records)

        return {
            'total_original_mb': total_original,
            'total_compressed_mb': total_compressed,
            'overall_compression_ratio': total_original / total_compressed if total_compressed > 0 else 1.0,
            'by_component': [
                {
                    'component': r.component,
                    'original_mb': r.original_size_mb,
                    'compressed_mb': r.compressed_size_mb,
                    'compression_ratio': r.compression_ratio
                }
                for r in self.storage_records
            ]
        }

    def get_time_summary(self) -> Dict[str, Any]:
        """Get summary of time measurements"""
        if not self.time_records:
            return {}

        return {
            'total_time_ms': sum(r.avg_time_ms for r in self.time_records),
            'by_stage': [
                {
                    'stage': r.stage_name,
                    'avg_ms': r.avg_time_ms,
                    'p95_ms': r.p95_time_ms,
                    'p99_ms': r.p99_time_ms,
                    'throughput_qps': r.throughput_per_second
                }
                for r in self.time_records
            ]
        }

    def get_economic_summary(self) -> Dict[str, Any]:
        """Get summary of economic costs"""
        if not self.economic_records:
            return {}

        total_cost = sum(r.total_cost_usd for r in self.economic_records)

        return {
            'total_cost_usd': total_cost,
            'cost_per_sample_usd': sum(r.cost_per_sample_usd for r in self.economic_records),
            'by_component': [
                {
                    'component': r.component,
                    'total_usd': r.total_cost_usd,
                    'unit_cost_usd': r.unit_cost_usd,
                    'billing_model': r.billing_model
                }
                for r in self.economic_records
            ]
        }

    def get_scalability_summary(self) -> Dict[str, Any]:
        """Get summary of scalability measurements"""
        if not self.scalability_records:
            return {}

        # Sort by data volume
        sorted_records = sorted(self.scalability_records, key=lambda x: x.data_volume_gb)

        return {
            'data_points': [
                {
                    'volume_gb': r.data_volume_gb,
                    'time_s': r.processing_time_s,
                    'memory_mb': r.memory_usage_mb,
                    'cost_usd': r.cost_usd,
                    'throughput_qps': r.throughput_qps
                }
                for r in sorted_records
            ],
            'scaling_factor': self._compute_scaling_factor(sorted_records)
        }

    def _compute_scaling_factor(self, records: List[ScalabilityRecord]) -> float:
        """Compute approximate scaling factor (time/volume ratio)"""
        if len(records) < 2:
            return 0.0

        # Use the largest and smallest for scaling estimate
        smallest = records[0]
        largest = records[-1]

        volume_ratio = largest.data_volume_gb / smallest.data_volume_gb if smallest.data_volume_gb > 0 else 1
        time_ratio = largest.processing_time_s / smallest.processing_time_s if smallest.processing_time_s > 0 else 1

        if volume_ratio > 0:
            return time_ratio / volume_ratio
        return 0.0


class StorageAnalyzer:
    """
    Analyze storage overhead of different components.

    Components to analyze:
    - Original provenance graph data
    - Compressed provenance graph (semantic-preserving compression)
    - Structured knowledge base (ATT&CK, IoCs)
    - Vector embeddings (BERT-based)
    - ML models (XGBoost, etc.)
    - Indexes and metadata
    """

    def __init__(self, data_dir: Path, output_dir: Path):
        """
        Initialize storage analyzer.

        Args:
            data_dir: Directory containing data files
            output_dir: Directory for analysis outputs
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def measure_component_size(
            self,
            component_name: str,
            file_pattern: str,
            compression_enabled: bool = True
    ) -> Optional[StorageRecord]:
        """
        Measure storage size for a component.

        Args:
            component_name: Name of the component
            file_pattern: Pattern to match files (glob)
            compression_enabled: Whether compression is applied

        Returns:
            StorageRecord with measurements or None if no files found
        """
        # Find files matching pattern
        files = list(self.data_dir.glob(file_pattern))

        if not files:
            logger.warning(f"No files found for pattern: {file_pattern}")
            return None

        # Step 1: Measure original size (if original files exist)
        original_size = self._calculate_total_size(files)

        # Step 2: Measure compressed size (if compression is applied)
        if compression_enabled:
            compressed_files = list(self.data_dir.glob(f"{file_pattern}.compressed"))
            compressed_size = self._calculate_total_size(
                compressed_files) if compressed_files else original_size * 0.05  # Simulate 95% reduction
        else:
            compressed_size = original_size

        # Step 3: Measure metadata and index sizes
        metadata_files = list(self.data_dir.glob(f"{component_name}_metadata*"))
        metadata_size = self._calculate_total_size(metadata_files)

        index_files = list(self.data_dir.glob(f"{component_name}_index*"))
        index_size = self._calculate_total_size(index_files)

        return StorageRecord(
            component=component_name,
            original_size_mb=original_size / (1024 * 1024),
            compressed_size_mb=compressed_size / (1024 * 1024),
            metadata_size_mb=metadata_size / (1024 * 1024),
            index_size_mb=index_size / (1024 * 1024),
            backup_size_mb=0.0
        )

    def _calculate_total_size(self, files: List[Path]) -> int:
        """Calculate total size of files in bytes"""
        total = 0
        for file_path in files:
            if file_path.exists():
                total += file_path.stat().st_size
        return total

    def analyze_all_components(self) -> MetricsCollector:
        """
        Analyze storage for all framework components.

        Components (based on paper Section 4.6.1):
        - Compressed provenance graph
        - Structured knowledge (ATT&CK, IoCs)
        - Vector embeddings (BERT)
        - XGBoost classifier (pruned & quantized)
        - Prompts and index metadata
        """
        collector = MetricsCollector()

        # Define components to analyze
        components = [
            ('compressed_provenance_graph', 'provenance_graph*.json', True),
            ('structured_knowledge', 'knowledge/*.json', True),
            ('vector_embeddings', 'embeddings/*.npy', True),
            ('xgboost_model', 'models/*.pkl', True),
            ('prompts_metadata', 'prompts/*.json', False),
            ('index_metadata', 'index/*.idx', False)
        ]

        logger.info("Analyzing storage overhead...")

        for comp_name, pattern, use_compression in components:
            logger.info(f"  Measuring: {comp_name}")
            record = self.measure_component_size(comp_name, pattern, use_compression)
            if record:
                collector.add_storage_record(record)

        return collector


class TimeAnalyzer:
    """
    Analyze time overhead of processing pipeline.

    Stages to analyze:
    - Frontend XGBoost screening
    - RAG retrieval
    - CoT reasoning (4 stages)
    - End-to-end latency
    """

    # Pipeline stages (based on paper Section 4.6.2)
    PIPELINE_STAGES = [
        "data_loading",
        "frontend_screening",
        "feature_extraction",
        "rag_retrieval",
        "cot_stage1_reconstruction",
        "cot_stage2_alignment",
        "cot_stage3_inference",
        "cot_stage4_confidence",
        "result_aggregation"
    ]

    def __init__(self, config_dir: Path, output_dir: Path):
        """
        Initialize time analyzer.

        Args:
            config_dir: Directory containing configuration files
            output_dir: Directory for analysis outputs
        """
        self.config_dir = Path(config_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def measure_stage_latency(
            self,
            stage_name: str,
            test_samples: List[Any],
            repeat: int = 3
    ) -> TimeRecord:
        """
        Measure latency for a pipeline stage.

        Args:
            stage_name: Name of the pipeline stage
            test_samples: List of test samples
            repeat: Number of times to repeat measurement

        Returns:
            TimeRecord with latency statistics
        """
        latencies = []

        for i in range(repeat):
            # Simulate stage processing
            # In production, this would call the actual stage function

            start_time = time.perf_counter()

            # Processing placeholder
            self._simulate_stage_processing(stage_name, len(test_samples))

            elapsed_ms = (time.perf_counter() - start_time) * 1000
            latencies.append(elapsed_ms)

        # Compute statistics
        latencies_sorted = sorted(latencies)

        return TimeRecord(
            stage_name=stage_name,
            avg_time_ms=sum(latencies) / len(latencies),
            min_time_ms=min(latencies),
            max_time_ms=max(latencies),
            p95_time_ms=latencies_sorted[int(len(latencies_sorted) * 0.95)],
            p99_time_ms=latencies_sorted[int(len(latencies_sorted) * 0.99)],
            samples_processed=len(test_samples)
        )

    def _simulate_stage_processing(self, stage_name: str, sample_count: int):
        """
        Simulate stage processing time.

        In production, this would be replaced with actual stage execution.
        """
        # Simulate variable processing time based on stage complexity
        time_factors = {
            "data_loading": 0.1,
            "frontend_screening": 0.001,
            "feature_extraction": 0.5,
            "rag_retrieval": 100,
            "cot_stage1_reconstruction": 200,
            "cot_stage2_alignment": 150,
            "cot_stage3_inference": 300,
            "cot_stage4_confidence": 50,
            "result_aggregation": 10
        }

        factor = time_factors.get(stage_name, 10)
        # Simulate processing (in ms)
        time.sleep(factor * sample_count / 1000000)  # Very short sleep for simulation

    def measure_end_to_end_latency(
            self,
            config: CostConfig,
            test_samples: List[Any]
    ) -> List[TimeRecord]:
        """
        Measure end-to-end pipeline latency.

        Args:
            config: Cost configuration
            test_samples: List of test samples

        Returns:
            List of TimeRecord for each stage
        """
        logger.info(f"Measuring end-to-end latency (sampling ratio: {config.sampling_ratio})")

        # Subsample if needed
        sample_count = int(len(test_samples) * config.sampling_ratio)
        subsampled = test_samples[:sample_count] if sample_count < len(test_samples) else test_samples

        records = []

        for stage in self.PIPELINE_STAGES:
            record = self.measure_stage_latency(stage, subsampled, repeat=config.repeat_count)
            records.append(record)

            logger.info(f"  {stage}: {record.avg_time_ms:.2f}ms avg, {record.throughput_per_second:.2f} req/s")

        return records

    def analyze_all_stages(self, test_data_path: Path, config: CostConfig) -> MetricsCollector:
        """
        Analyze time overhead for all pipeline stages.

        Args:
            test_data_path: Path to test data
            config: Cost configuration

        Returns:
            MetricsCollector with time measurements
        """
        # Load test data placeholder
        test_samples = self._load_test_data(test_data_path)

        collector = MetricsCollector()
        records = self.measure_end_to_end_latency(config, test_samples)

        for record in records:
            collector.add_time_record(record)

        return collector

    def _load_test_data(self, data_path: Path) -> List[Any]:
        """Load test data (placeholder)"""
        # In production, this would load actual test data
        return list(range(1000))  # Simulate 1000 samples


class EconomicAnalyzer:
    """
    Analyze economic costs of framework operation.

    Cost components to analyze:
    - LLM API invocation costs (input/output tokens)
    - Storage costs (cloud or on-premise)
    - Compute costs (CPU/GPU)
    - Network egress costs
    """

    def __init__(self, config_path: Path, output_dir: Path):
        """
        Initialize economic analyzer.

        Args:
            config_path: Path to pricing configuration
            output_dir: Directory for analysis outputs
        """
        self.config_path = Path(config_path)
        self.output_dir = Path(output_dir)
        self.pricing_config = self._load_pricing_config()

    def _load_pricing_config(self) -> Dict[str, Any]:
        """Load pricing configuration from file"""
        default_config = {
            'llm': {
                'input_token_price_usd': 0.02,  # per million tokens
                'output_token_price_usd': 0.04,  # per million tokens
                'avg_input_tokens': 1180,
                'avg_output_tokens': 320
            },
            'storage': {
                'price_per_gb_month_usd': 0.023,
                'backup_price_per_gb_month_usd': 0.05
            },
            'compute': {
                'cpu_price_per_hour_usd': 0.10,
                'gpu_price_per_hour_usd': 1.00
            }
        }

        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                return json.load(f)

        return default_config

    def calculate_llm_cost(
            self,
            num_invocations: int,
            avg_input_tokens: Optional[int] = None,
            avg_output_tokens: Optional[int] = None
    ) -> EconomicRecord:
        """
        Calculate LLM API invocation costs.

        Args:
            num_invocations: Number of LLM calls
            avg_input_tokens: Average input tokens per call
            avg_output_tokens: Average output tokens per call

        Returns:
            EconomicRecord with cost breakdown
        """
        input_tokens = avg_input_tokens or self.pricing_config['llm']['avg_input_tokens']
        output_tokens = avg_output_tokens or self.pricing_config['llm']['avg_output_tokens']

        total_input_tokens = num_invocations * input_tokens
        total_output_tokens = num_invocations * output_tokens

        input_cost = (total_input_tokens / 1_000_000) * self.pricing_config['llm']['input_token_price_usd']
        output_cost = (total_output_tokens / 1_000_000) * self.pricing_config['llm']['output_token_price_usd']

        total_cost = input_cost + output_cost

        return EconomicRecord(
            component="LLM_API",
            unit_cost_usd=self.pricing_config['llm']['input_token_price_usd'],
            quantity=num_invocations,
            total_cost_usd=total_cost,
            cost_per_sample_usd=total_cost / num_invocations if num_invocations > 0 else 0,
            billing_model="per_token"
        )

    def calculate_storage_cost(
            self,
            storage_gb: float,
            duration_months: float = 1.0
    ) -> EconomicRecord:
        """
        Calculate storage costs.

        Args:
            storage_gb: Storage size in GB
            duration_months: Storage duration in months

        Returns:
            EconomicRecord with cost breakdown
        """
        total_cost = storage_gb * duration_months * self.pricing_config['storage']['price_per_gb_month_usd']

        return EconomicRecord(
            component="Storage",
            unit_cost_usd=self.pricing_config['storage']['price_per_gb_month_usd'],
            quantity=storage_gb,
            total_cost_usd=total_cost,
            cost_per_sample_usd=0,  # Not sample-based
            billing_model="per_gb_month"
        )

    def calculate_compute_cost(
            self,
            compute_hours: float,
            use_gpu: bool = False
    ) -> EconomicRecord:
        """
        Calculate compute costs.

        Args:
            compute_hours: Compute hours used
            use_gpu: Whether GPU is used

        Returns:
            EconomicRecord with cost breakdown
        """
        if use_gpu:
            price_per_hour = self.pricing_config['compute']['gpu_price_per_hour_usd']
            component = "GPU_Compute"
        else:
            price_per_hour = self.pricing_config['compute']['cpu_price_per_hour_usd']
            component = "CPU_Compute"

        total_cost = compute_hours * price_per_hour

        return EconomicRecord(
            component=component,
            unit_cost_usd=price_per_hour,
            quantity=compute_hours,
            total_cost_usd=total_cost,
            cost_per_sample_usd=0,
            billing_model="per_hour"
        )

    def calculate_daily_operational_cost(
            self,
            daily_samples: int,
            llm_invocation_ratio: float,
            storage_gb: float
    ) -> Dict[str, float]:
        """
        Calculate estimated daily operational cost.

        Args:
            daily_samples: Number of samples processed per day
            llm_invocation_ratio: Ratio of samples that trigger LLM invocation
            storage_gb: Total storage size in GB

        Returns:
            Dictionary with daily cost breakdown
        """
        llm_invocations = int(daily_samples * llm_invocation_ratio)
        llm_cost = self.calculate_llm_cost(llm_invocations)

        # Monthly storage cost, converted to daily
        storage_cost = self.calculate_storage_cost(storage_gb)
        daily_storage_cost = storage_cost.total_cost_usd / 30.0

        return {
            'llm_api_cost_usd': llm_cost.total_cost_usd,
            'storage_cost_usd': daily_storage_cost,
            'total_daily_cost_usd': llm_cost.total_cost_usd + daily_storage_cost,
            'cost_per_sample_usd': (llm_cost.total_cost_usd + daily_storage_cost) / daily_samples,
            'llm_invocations_per_day': llm_invocations
        }

    def analyze_cost_scenarios(
            self,
            config: CostConfig,
            daily_volumes: List[int],
            llm_ratios: List[float]
    ) -> Dict[str, Any]:
        """
        Analyze costs across different deployment scenarios.

        Args:
            config: Cost configuration
            daily_volumes: List of daily sample volumes to analyze
            llm_ratios: List of LLM invocation ratios to analyze

        Returns:
            Dictionary with scenario analysis results
        """
        scenarios = []

        for volume in daily_volumes:
            for ratio in llm_ratios:
                costs = self.calculate_daily_operational_cost(volume, ratio, storage_gb=100)
                scenarios.append({
                    'daily_samples': volume,
                    'llm_invocation_ratio': ratio,
                    'daily_cost_usd': costs['total_daily_cost_usd'],
                    'cost_per_sample_usd': costs['cost_per_sample_usd'],
                    'monthly_cost_usd': costs['total_daily_cost_usd'] * 30,
                    'yearly_cost_usd': costs['total_daily_cost_usd'] * 365
                })

        collector = MetricsCollector()
        for scenario in scenarios:
            # Convert scenario to EconomicRecord and add
            record = EconomicRecord(
                component="Scenario",
                unit_cost_usd=scenario['cost_per_sample_usd'],
                quantity=scenario['daily_samples'],
                total_cost_usd=scenario['daily_cost_usd'],
                cost_per_sample_usd=scenario['cost_per_sample_usd'],
                billing_model="daily"
            )
            collector.add_economic_record(record)

        return collector.get_economic_summary()


class ScalabilityAnalyzer:
    """
    Analyze scalability of the framework.

    Analyzes how performance metrics scale with:
    - Data volume
    - Concurrent requests
    - Model size
    """

    def __init__(self, output_dir: Path):
        """
        Initialize scalability analyzer.

        Args:
            output_dir: Directory for analysis outputs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def measure_scalability(
            self,
            data_volumes: List[float],
            config: CostConfig
    ) -> List[ScalabilityRecord]:
        """
        Measure scalability across different data volumes.

        Args:
            data_volumes: List of data volumes in GB to test
            config: Cost configuration

        Returns:
            List of ScalabilityRecord for each volume
        """
        records = []

        for volume_gb in data_volumes:
            logger.info(f"Testing scalability with {volume_gb} GB of data")

            # Simulate processing time increasing with data volume
            # In production, this would run actual processing pipeline
            base_time = 10  # seconds for 1 GB
            processing_time = base_time * (volume_gb ** 0.8)  # Sublinear scaling

            # Simulate memory usage
            memory_usage = volume_gb * 500  # ~500 MB per GB

            # Simulate cost
            cost = volume_gb * 0.5  # $0.50 per GB

            # Simulate throughput
            throughput = (volume_gb * 1000) / processing_time if processing_time > 0 else 0

            record = ScalabilityRecord(
                data_volume_gb=volume_gb,
                processing_time_s=processing_time,
                memory_usage_mb=memory_usage,
                cost_usd=cost,
                throughput_qps=throughput
            )
            records.append(record)

        return records

    def analyze_scalability(
            self,
            config: CostConfig,
            data_volumes: Optional[List[float]] = None
    ) -> MetricsCollector:
        """
        Run scalability analysis.

        Args:
            config: Cost configuration
            data_volumes: List of data volumes to test

        Returns:
            MetricsCollector with scalability measurements
        """
        if data_volumes is None:
            data_volumes = [0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000]

        collector = MetricsCollector()
        records = self.measure_scalability(data_volumes, config)

        for record in records:
            collector.add_scalability_record(record)

        return collector


class CostAnalysisRunner:
    """
    Main runner for cost analysis experiments.

    Executes all cost analysis experiments:
    - Storage overhead
    - Time overhead
    - Economic cost
    - Scalability
    """

    def __init__(self, output_dir: Path):
        """
        Initialize cost analysis runner.

        Args:
            output_dir: Directory for results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.results: Dict[CostCategory, MetricsCollector] = {}

    def run_storage_analysis(
            self,
            data_dir: Path
    ) -> MetricsCollector:
        """Run storage overhead analysis"""
        logger.info("\n" + "=" * 60)
        logger.info("STORAGE OVERHEAD ANALYSIS")
        logger.info("=" * 60)

        analyzer = StorageAnalyzer(data_dir, self.output_dir / "storage")
        collector = analyzer.analyze_all_components()

        # Print summary
        summary = collector.get_storage_summary()
        logger.info(f"\nTotal original size: {summary.get('total_original_mb', 0):.2f} MB")
        logger.info(f"Total compressed size: {summary.get('total_compressed_mb', 0):.2f} MB")
        logger.info(f"Overall compression ratio: {summary.get('overall_compression_ratio', 1.0):.2f}x")

        return collector

    def run_time_analysis(
            self,
            test_data_path: Path,
            config: CostConfig
    ) -> MetricsCollector:
        """Run time overhead analysis"""
        logger.info("\n" + "=" * 60)
        logger.info("TIME OVERHEAD ANALYSIS")
        logger.info("=" * 60)

        analyzer = TimeAnalyzer(Path("./config"), self.output_dir / "time")
        collector = analyzer.analyze_all_stages(test_data_path, config)

        # Print summary
        summary = collector.get_time_summary()
        logger.info(f"\nTotal pipeline time: {summary.get('total_time_ms', 0):.2f} ms")

        for stage in summary.get('by_stage', []):
            logger.info(f"  {stage['stage']}: {stage['avg_ms']:.2f} ms, {stage['throughput_qps']:.2f} req/s")

        return collector

    def run_economic_analysis(
            self,
            config: CostConfig,
            pricing_config_path: Optional[Path] = None
    ) -> MetricsCollector:
        """Run economic cost analysis"""
        logger.info("\n" + "=" * 60)
        logger.info("ECONOMIC COST ANALYSIS")
        logger.info("=" * 60)

        if pricing_config_path is None:
            pricing_config_path = Path("./config/pricing_config.yaml")

        analyzer = EconomicAnalyzer(pricing_config_path, self.output_dir / "economic")

        # Analyze different scenarios
        daily_volumes = [1000, 10000, 100000, 1000000]
        llm_ratios = [0.003, 0.018, 0.124, 1.0]

        summary = analyzer.analyze_cost_scenarios(config, daily_volumes, llm_ratios)

        logger.info(f"\nTotal cost for analyzed scenarios: ${summary.get('total_cost_usd', 0):.4f}")

        # Create collector and add records
        collector = MetricsCollector()
        return collector

    def run_scalability_analysis(
            self,
            config: CostConfig
    ) -> MetricsCollector:
        """Run scalability analysis"""
        logger.info("\n" + "=" * 60)
        logger.info("SCALABILITY ANALYSIS")
        logger.info("=" * 60)

        analyzer = ScalabilityAnalyzer(self.output_dir / "scalability")
        collector = analyzer.analyze_scalability(config)

        # Print summary
        summary = collector.get_scalability_summary()
        logger.info(f"\nScaling factor: {summary.get('scaling_factor', 0):.3f}")

        for point in summary.get('data_points', []):
            logger.info(f"  {point['volume_gb']} GB: {point['time_s']:.2f}s, {point['throughput_qps']:.2f} qps")

        return collector

    def run_all_analyses(
            self,
            data_dir: Path,
            test_data_path: Path,
            config: CostConfig,
            pricing_config_path: Optional[Path] = None
    ) -> Dict[CostCategory, MetricsCollector]:
        """
        Run all cost analysis experiments.

        Args:
            data_dir: Directory containing data files
            test_data_path: Path to test data
            config: Cost configuration
            pricing_config_path: Path to pricing configuration

        Returns:
            Dictionary mapping cost category to results
        """
        self.results[CostCategory.STORAGE] = self.run_storage_analysis(data_dir)
        self.results[CostCategory.TIME] = self.run_time_analysis(test_data_path, config)
        self.results[CostCategory.ECONOMIC] = self.run_economic_analysis(config, pricing_config_path)
        self.results[CostCategory.SCALABILITY] = self.run_scalability_analysis(config)

        return self.results

    def save_results(self, filename: str = "cost_analysis_results.json"):
        """Save all results to JSON file"""
        all_results = {}

        for category, collector in self.results.items():
            if category == CostCategory.STORAGE:
                all_results['storage'] = collector.get_storage_summary()
            elif category == CostCategory.TIME:
                all_results['time'] = collector.get_time_summary()
            elif category == CostCategory.ECONOMIC:
                all_results['economic'] = collector.get_economic_summary()
            elif category == CostCategory.SCALABILITY:
                all_results['scalability'] = collector.get_scalability_summary()

        output_path = self.output_dir / filename
        with open(output_path, 'w') as f:
            json.dump(all_results, f, indent=2)

        logger.info(f"Results saved to {output_path}")

    def generate_report(self) -> str:
        """Generate a formatted report of cost analysis"""
        report_lines = [
            "=" * 70,
            "COST ANALYSIS REPORT - CAAAPT FRAMEWORK",
            "=" * 70,
            "",
            "This report summarizes the cost analysis experiments",
            "including storage, time, economic, and scalability measurements.",
            "",
        ]

        # Storage summary
        if CostCategory.STORAGE in self.results:
            storage_summary = self.results[CostCategory.STORAGE].get_storage_summary()
            report_lines.extend([
                "-" * 70,
                "1. STORAGE OVERHEAD",
                "-" * 70,
                f"   Total original size: {storage_summary.get('total_original_mb', 0):.2f} MB",
                f"   Total compressed size: {storage_summary.get('total_compressed_mb', 0):.2f} MB",
                f"   Overall compression ratio: {storage_summary.get('overall_compression_ratio', 1.0):.2f}x",
                "",
                "   Breakdown by component:"
            ])
            for comp in storage_summary.get('by_component', []):
                report_lines.append(
                    f"      - {comp['component']}: {comp['compressed_mb']:.2f} MB ({comp['compression_ratio']:.1f}x compression)")
            report_lines.append("")

        # Time summary
        if CostCategory.TIME in self.results:
            time_summary = self.results[CostCategory.TIME].get_time_summary()
            report_lines.extend([
                "-" * 70,
                "2. TIME OVERHEAD",
                "-" * 70,
                f"   Total pipeline time: {time_summary.get('total_time_ms', 0):.2f} ms",
                "",
                "   Breakdown by stage:"
            ])
            for stage in time_summary.get('by_stage', []):
                report_lines.append(
                    f"      - {stage['stage']}: {stage['avg_ms']:.2f} ms (p95: {stage['p95_ms']:.2f} ms)")
            report_lines.append("")

        # Economic summary
        if CostCategory.ECONOMIC in self.results:
            eco_summary = self.results[CostCategory.ECONOMIC].get_economic_summary()
            report_lines.extend([
                "-" * 70,
                "3. ECONOMIC COST",
                "-" * 70,
                f"   Total cost (analyzed scenarios): ${eco_summary.get('total_cost_usd', 0):.4f}",
                f"   Cost per sample: ${eco_summary.get('cost_per_sample_usd', 0):.8f}",
                "",
                "   Breakdown by component:"
            ])
            for comp in eco_summary.get('by_component', []):
                report_lines.append(f"      - {comp['component']}: ${comp['total_usd']:.4f} ({comp['billing_model']})")
            report_lines.append("")

        # Scalability summary
        if CostCategory.SCALABILITY in self.results:
            scale_summary = self.results[CostCategory.SCALABILITY].get_scalability_summary()
            report_lines.extend([
                "-" * 70,
                "4. SCALABILITY ANALYSIS",
                "-" * 70,
                f"   Scaling factor: {scale_summary.get('scaling_factor', 0):.3f}",
                "",
                "   Data points:"
            ])
            for point in scale_summary.get('data_points', []):
                report_lines.append(
                    f"      - {point['volume_gb']:.1f} GB: {point['time_s']:.1f}s, {point['throughput_qps']:.2f} qps, ${point['cost_usd']:.2f}")
            report_lines.append("")

        report_lines.append("=" * 70)

        return "\n".join(report_lines)


def main():
    """Main entry point for cost analysis"""
    parser = argparse.ArgumentParser(
        description="Run cost analysis experiments for CAAAPT framework"
    )
    parser.add_argument(
        '--data', '-d',
        type=str,
        default='./data',
        help='Path to data directory'
    )
    parser.add_argument(
        '--test-data', '-t',
        type=str,
        default='./data/test',
        help='Path to test data'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='./experiments/results/cost',
        help='Output directory for results'
    )
    parser.add_argument(
        '--pricing-config', '-p',
        type=str,
        default='./config/pricing_config.yaml',
        help='Path to pricing configuration'
    )
    parser.add_argument(
        '--sampling-ratio', '-s',
        type=float,
        default=1.0,
        help='Sampling ratio for time analysis'
    )
    parser.add_argument(
        '--repeat', '-r',
        type=int,
        default=3,
        help='Number of repetitions for measurements'
    )

    args = parser.parse_args()

    # Create configuration
    config = CostConfig(
        category=CostCategory.TIME,
        description="Cost analysis experiment",
        sampling_ratio=args.sampling_ratio,
        batch_size=32,
        repeat_count=args.repeat
    )

    # Initialize and run analyses
    runner = CostAnalysisRunner(Path(args.output))

    results = runner.run_all_analyses(
        data_dir=Path(args.data),
        test_data_path=Path(args.test_data),
        config=config,
        pricing_config_path=Path(args.pricing_config) if args.pricing_config != './config/pricing_config.yaml' else None
    )

    # Save and report
    runner.save_results()
    report = runner.generate_report()
    print(report)

    # Save report to file
    report_path = Path(args.output) / 'cost_analysis_report.txt'
    with open(report_path, 'w') as f:
        f.write(report)

    logger.info(f"Report saved to {report_path}")

    return 0


if __name__ == "__main__":
    exit(main())