
"""
Ablation Study Module for CAAAPT Framework

This module implements the ablation experiment pipeline to evaluate the contribution
of each core component in the framework. The code is fully desensitized and only
contains the experimental workflow without any actual data or result values.

Experiment Configurations (based on paper Section 4.7):
- Configuration A: Full framework (all components enabled)
- Configuration B: Remove frontend screening component
- Configuration C: Remove RAG knowledge retrieval component
- Configuration D: Remove CoT reasoning component

"""

import argparse
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AblationMode(Enum):
    """Ablation experiment modes"""
    FULL = "full"  # All components enabled
    WITHOUT_XGBOOST = "without_xgboost"  # Remove frontend screening
    WITHOUT_RAG = "without_rag"  # Remove RAG retrieval
    WITHOUT_COT = "without_cot"  # Remove CoT reasoning


@dataclass
class ExperimentConfig:
    """Configuration for a single ablation experiment"""
    mode: AblationMode
    description: str
    enable_frontend: bool = True
    enable_rag: bool = True
    enable_cot: bool = True
    num_trials: int = 3  # Number of repeated trials for statistical significance

    @classmethod
    def from_mode(cls, mode: AblationMode) -> 'ExperimentConfig':
        """Create configuration from ablation mode"""
        configs = {
            AblationMode.FULL: cls(
                mode=mode,
                description="Full framework with all components",
                enable_frontend=True,
                enable_rag=True,
                enable_cot=True
            ),
            AblationMode.WITHOUT_XGBOOST: cls(
                mode=mode,
                description="Remove frontend XGBoost screening",
                enable_frontend=False,
                enable_rag=True,
                enable_cot=True
            ),
            AblationMode.WITHOUT_RAG: cls(
                mode=mode,
                description="Remove RAG knowledge retrieval",
                enable_frontend=True,
                enable_rag=False,
                enable_cot=True
            ),
            AblationMode.WITHOUT_COT: cls(
                mode=mode,
                description="Remove CoT reasoning",
                enable_frontend=True,
                enable_rag=True,
                enable_cot=False
            )
        }
        return configs[mode]


@dataclass
class ExperimentResult:
    """Store results of a single ablation experiment"""
    config: ExperimentConfig
    trial_id: int
    metrics: Dict[str, float]
    latency_seconds: float
    resource_usage_mb: float
    error_log: Optional[str] = None


class DataLoader:
    """Handle data loading for ablation experiments"""

    def __init__(self, data_path: Path, validation_split: float = 0.2):
        """
        Initialize data loader.

        Args:
            data_path: Path to the dataset directory
            validation_split: Fraction of data to use for validation
        """
        self.data_path = Path(data_path)
        self.validation_split = validation_split
        self.train_data = None
        self.val_data = None
        self.test_data = None

    def load(self) -> Dict[str, Any]:
        """
        Load and split the dataset.

        Returns:
            Dictionary containing train/val/test splits
        """
        logger.info(f"Loading data from {self.data_path}")

        # Step 1: Discover data files
        # The dataset should be prepared by preprocess_trec.py or similar
        # Expecting format compatible with the framework's input requirements

        # Placeholder for actual data loading logic
        # In production, this would load the processed dataset files

        logger.info("Data loading complete")
        return {
            'train': self.train_data,
            'val': self.val_data,
            'test': self.test_data
        }


class ModelManager:
    """Manage model loading and component toggling for ablation"""

    def __init__(self, model_dir: Path):
        """
        Initialize model manager.

        Args:
            model_dir: Directory containing trained models
        """
        self.model_dir = Path(model_dir)
        self.frontend_model = None
        self.backend_model = None
        self.retriever = None

    def load_models(self, config: ExperimentConfig) -> None:
        """
        Load models according to experiment configuration.

        Args:
            config: Experiment configuration specifying which components to load
        """
        logger.info(f"Loading models for config: {config.mode.value}")

        # Step 1: Load frontend XGBoost model if enabled
        if config.enable_frontend:
            # Load pre-trained XGBoost model from model_dir
            # Expected file: xgboost_screener.pkl or xgboost_screener.json
            logger.info("  - Loading XGBoost frontend screener")
            pass
        else:
            logger.info("  - Skipping XGBoost frontend (disabled for ablation)")

        # Step 2: Load backend LLM components
        if config.enable_rag:
            # Initialize RAG retriever with knowledge base
            logger.info("  - Loading RAG retriever with knowledge base")
            pass
        else:
            logger.info("  - Skipping RAG retriever (disabled for ablation)")

        if config.enable_cot:
            # Load CoT prompt templates
            logger.info("  - Loading CoT reasoning templates")
            pass
        else:
            logger.info("  - Skipping CoT reasoning (disabled for ablation)")

    def get_component_status(self) -> Dict[str, bool]:
        """Get current status of all components"""
        return {
            'frontend': self.frontend_model is not None,
            'rag': self.retriever is not None,
            'cot': True  # Placeholder
        }


class MetricsCollector:
    """
    Collect and compute evaluation metrics for ablation experiments.

    Metrics to collect (as described in paper Section 4.2):
    - Binary classification: F1-score, Precision, Recall
    - Multi-class classification: ACC, Top3ACC, TacticACC
    - Performance: Latency, Throughput
    - Resource: Memory usage, Computational cost
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all collected metrics"""
        self.predictions = []
        self.ground_truth = []
        self.inference_times = []
        self.confidence_scores = []

    def record_prediction(self, pred: str, true: str, time_ms: float, confidence: float):
        """Record a single prediction"""
        self.predictions.append(pred)
        self.ground_truth.append(true)
        self.inference_times.append(time_ms)
        self.confidence_scores.append(confidence)

    def compute_metrics(self) -> Dict[str, float]:
        """
        Compute all evaluation metrics.

        Returns:
            Dictionary containing all computed metrics
        """
        metrics = {}

        # Step 1: Compute binary classification metrics
        # (for benign vs malicious detection)
        metrics.update(self._compute_binary_metrics())

        # Step 2: Compute multi-class classification metrics
        # (for tactic/technique recognition)
        metrics.update(self._compute_multiclass_metrics())

        # Step 3: Compute performance metrics
        metrics.update(self._compute_performance_metrics())

        return metrics

    def _compute_binary_metrics(self) -> Dict[str, float]:
        """
        Compute binary classification metrics.

        Returns:
            Dictionary with precision, recall, f1, accuracy
        """
        # Placeholder for actual metric computation
        # In production, use sklearn.metrics or custom implementation
        return {
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'binary_accuracy': 0.0
        }

    def _compute_multiclass_metrics(self) -> Dict[str, float]:
        """
        Compute multi-class classification metrics.

        Metrics (paper Section 4.2):
        - ACC: Top-1 accuracy
        - Top3ACC: Top-3 accuracy
        - TacticACC: Tactic-level accuracy

        Returns:
            Dictionary with multiclass metrics
        """
        return {
            'top1_accuracy': 0.0,
            'top3_accuracy': 0.0,
            'tactic_accuracy': 0.0,
            'macro_f1': 0.0,
            'weighted_f1': 0.0
        }

    def _compute_performance_metrics(self) -> Dict[str, float]:
        """
        Compute performance metrics.

        Returns:
            Dictionary with latency and throughput metrics
        """
        if self.inference_times:
            avg_latency = sum(self.inference_times) / len(self.inference_times)
        else:
            avg_latency = 0.0

        return {
            'avg_latency_ms': avg_latency,
            'p95_latency_ms': 0.0,
            'p99_latency_ms': 0.0,
            'throughput_qps': 0.0
        }


class AblationExperimentRunner:
    """
    Main runner for ablation experiments.

    Executes the experiment pipeline:
    1. Load dataset
    2. Configure models based on ablation mode
    3. Run inference on test set
    4. Collect metrics
    5. Repeat multiple trials
    6. Aggregate and report results
    """

    def __init__(self, output_dir: Path):
        """
        Initialize experiment runner.

        Args:
            output_dir: Directory to save experiment results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.results: Dict[AblationMode, List[ExperimentResult]] = {}

    def run_experiment(
            self,
            config: ExperimentConfig,
            data_loader: DataLoader,
            model_manager: ModelManager,
            trial_id: int
    ) -> ExperimentResult:
        """
        Run a single ablation experiment.

        Args:
            config: Experiment configuration
            data_loader: Data loader instance
            model_manager: Model manager instance
            trial_id: Trial identifier for repetition

        Returns:
            ExperimentResult with metrics and metadata
        """
        logger.info(f"Running trial {trial_id} for config: {config.mode.value}")

        # Step 1: Load test data
        data = data_loader.load()
        test_samples = data.get('test', [])

        # Step 2: Load models according to configuration
        model_manager.load_models(config)

        # Step 3: Initialize metrics collector
        metrics_collector = MetricsCollector()

        # Step 4: Run inference pipeline
        start_time = time.time()

        # Placeholder for actual inference logic
        # for sample in test_samples:
        #     # Frontend screening if enabled
        #     if config.enable_frontend:
        #         score = model_manager.frontend_model.predict(sample)
        #         if score < threshold:
        #             continue  # Filtered out as benign
        #
        #     # RAG retrieval if enabled
        #     context = None
        #     if config.enable_rag:
        #         context = model_manager.retriever.retrieve(sample)
        #
        #     # LLM inference with/without CoT
        #     result = model_manager.backend_model.infer(sample, context, use_cot=config.enable_cot)
        #
        #     # Record metrics
        #     metrics_collector.record_prediction(
        #         pred=result.predicted_label,
        #         true=sample.ground_truth,
        #         time_ms=result.inference_time_ms,
        #         confidence=result.confidence
        #     )

        total_time = time.time() - start_time

        # Step 5: Compute metrics
        metrics = metrics_collector.compute_metrics()

        # Step 6: Track resource usage (optional)
        resource_usage = self._get_memory_usage()

        return ExperimentResult(
            config=config,
            trial_id=trial_id,
            metrics=metrics,
            latency_seconds=total_time,
            resource_usage_mb=resource_usage
        )

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024

    def run_all_ablations(
            self,
            data_path: Path,
            model_dir: Path,
            num_trials: int = 3,
            modes: Optional[List[AblationMode]] = None
    ) -> Dict[AblationMode, List[ExperimentResult]]:
        """
        Run all ablation experiments.

        Args:
            data_path: Path to dataset directory
            model_dir: Directory containing trained models
            num_trials: Number of repeated trials per configuration
            modes: List of ablation modes to run (default: all)

        Returns:
            Dictionary mapping ablation mode to list of results
        """
        if modes is None:
            modes = list(AblationMode)

        # Initialize components
        data_loader = DataLoader(data_path)
        model_manager = ModelManager(model_dir)

        results = {}

        for mode in modes:
            logger.info(f"\n{'=' * 60}")
            logger.info(f"Starting ablation: {mode.value}")
            logger.info(f"{'=' * 60}")

            config = ExperimentConfig.from_mode(mode)
            config.num_trials = num_trials

            mode_results = []

            for trial in range(num_trials):
                logger.info(f"\n--- Trial {trial + 1}/{num_trials} ---")

                result = self.run_experiment(
                    config=config,
                    data_loader=data_loader,
                    model_manager=model_manager,
                    trial_id=trial + 1
                )
                mode_results.append(result)

                # Log trial results
                logger.info(f"Trial {trial + 1} completed. Metrics: {result.metrics}")

            results[mode] = mode_results

        self.results = results
        return results

    def aggregate_results(self) -> Dict[str, Any]:
        """
        Aggregate results across trials for each configuration.

        Returns:
            Dictionary with mean and std for each metric per configuration
        """
        aggregated = {}

        for mode, mode_results in self.results.items():
            # Collect metrics across trials
            all_metrics = [r.metrics for r in mode_results]

            # Compute mean and std for each metric
            metric_names = all_metrics[0].keys() if all_metrics else []
            means = {}
            stds = {}

            for metric_name in metric_names:
                values = [m[metric_name] for m in all_metrics]
                means[metric_name] = sum(values) / len(values)
                stds[metric_name] = self._compute_std(values)

            aggregated[mode.value] = {
                'mean_metrics': means,
                'std_metrics': stds,
                'num_trials': len(mode_results),
                'mean_latency': sum(r.latency_seconds for r in mode_results) / len(mode_results),
                'mean_resource_mb': sum(r.resource_usage_mb for r in mode_results) / len(mode_results)
            }

        return aggregated

    def _compute_std(self, values: List[float]) -> float:
        """Compute standard deviation"""
        if len(values) <= 1:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return variance ** 0.5

    def save_results(self, filename: str = "ablation_results.json"):
        """Save experiment results to file"""
        import json

        aggregated = self.aggregate_results()
        output_path = self.output_dir / filename

        with open(output_path, 'w') as f:
            json.dump(aggregated, f, indent=2)

        logger.info(f"Results saved to {output_path}")

    def generate_report(self) -> str:
        """
        Generate a text report of ablation results.

        Returns:
            Formatted report string
        """
        aggregated = self.aggregate_results()

        report_lines = [
            "=" * 70,
            "ABLATION STUDY RESULTS",
            "=" * 70,
            "",
            "This report shows the contribution of each framework component",
            "as measured by the difference in performance metrics.",
            "",
            "-" * 70,
            "Configuration Summary",
            "-" * 70,
        ]

        for mode_name, stats in aggregated.items():
            report_lines.append(f"\n{mode_name.upper()}:")
            report_lines.append(f"  Trials: {stats['num_trials']}")
            report_lines.append(f"  Avg Latency: {stats['mean_latency']:.3f}s")
            report_lines.append(f"  Memory Usage: {stats['mean_resource_mb']:.1f}MB")
            report_lines.append("  Metrics:")

            for metric, value in stats['mean_metrics'].items():
                std = stats['std_metrics'].get(metric, 0)
                report_lines.append(f"    {metric}: {value:.4f} ± {std:.4f}")

        # Compute component contributions (difference from full config)
        if AblationMode.FULL.value in aggregated:
            full_metrics = aggregated[AblationMode.FULL.value]['mean_metrics']

            report_lines.extend([
                "",
                "-" * 70,
                "Component Contributions (Delta from Full Framework)",
                "-" * 70,
            ])

            for mode in [AblationMode.WITHOUT_XGBOOST, AblationMode.WITHOUT_RAG, AblationMode.WITHOUT_COT]:
                if mode.value in aggregated:
                    mode_metrics = aggregated[mode.value]['mean_metrics']
                    deltas = {}

                    for metric in full_metrics:
                        if metric in mode_metrics:
                            deltas[metric] = full_metrics[metric] - mode_metrics[metric]

                    report_lines.append(f"\n{mode.value.replace('_', ' ').title()}:")
                    for metric, delta in deltas.items():
                        sign = "+" if delta > 0 else ""
                        report_lines.append(f"    {metric}: {sign}{delta:.4f}")

        report_lines.append("\n" + "=" * 70)

        return "\n".join(report_lines)


def main():
    """Main entry point for ablation experiments"""
    parser = argparse.ArgumentParser(
        description="Run ablation experiments for CAAAPT framework"
    )
    parser.add_argument(
        '--data', '-d',
        type=str,
        required=True,
        help='Path to dataset directory'
    )
    parser.add_argument(
        '--models', '-m',
        type=str,
        required=True,
        help='Path to trained models directory'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='./experiments/results/ablation',
        help='Output directory for results'
    )
    parser.add_argument(
        '--trials', '-t',
        type=int,
        default=3,
        help='Number of repeated trials per configuration'
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['full', 'without_xgboost', 'without_rag', 'without_cot', 'all'],
        default='all',
        help='Which ablation configuration to run'
    )

    args = parser.parse_args()

    # Determine which modes to run
    if args.mode == 'all':
        modes = list(AblationMode)
    else:
        modes = [AblationMode(args.mode)]

    # Initialize and run experiments
    runner = AblationExperimentRunner(Path(args.output))

    results = runner.run_all_ablations(
        data_path=Path(args.data),
        model_dir=Path(args.models),
        num_trials=args.trials,
        modes=modes
    )

    # Save and report
    runner.save_results()
    report = runner.generate_report()
    print(report)

    # Save report to file
    report_path = Path(args.output) / 'ablation_report.txt'
    with open(report_path, 'w') as f:
        f.write(report)

    logger.info(f"Report saved to {report_path}")

    return 0


if __name__ == "__main__":
    exit(main())