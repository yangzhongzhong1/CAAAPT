
import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Generator
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading
import logging
from contextlib import contextmanager

# Database imports (adjust based on your DB)
try:
    from sqlalchemy import create_engine, Column, String, Float, Integer, DateTime, JSON, Index
    from sqlalchemy.ext.declarative import declarative_base
    from sqlalchemy.orm import sessionmaker, Session
    from sqlalchemy.pool import QueuePool

    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False
    logging.warning("SQLAlchemy not available. Database features disabled.")

# Redis for real-time caching
try:
    import redis

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

# Kafka for streaming
try:
    from kafka import KafkaConsumer, KafkaProducer

    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False

from prometheus_client import Counter, Histogram, Gauge
import mlflow
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)


class ParetoOperationMode(Enum):
    """Operating modes for Pareto analyzer"""
    BATCH = "batch"  # Historical analysis
    STREAMING = "streaming"  # Real-time updates
    HYBRID = "hybrid"  # Both batch and streaming


class CostStrategy(Enum):
    """Cost calculation strategies"""
    COMPUTE_ONLY = "compute"
    API_ONLY = "api"
    HYBRID_COST = "hybrid"


@dataclass
class ParetoPoint:
    """Represents a single point on the Pareto frontier with production metadata"""
    tau_front: float
    tau_back: float
    cost: float
    accuracy: float
    f1_score: float
    precision: float
    recall: float
    logs_processed_pct: float

    # Production extensions
    timestamp: datetime = field(default_factory=datetime.now)
    experiment_id: Optional[str] = None
    run_id: Optional[str] = None
    model_version: Optional[str] = None
    environment: str = "development"  # dev/staging/production

    # Resource metrics
    cpu_usage_ms: Optional[float] = None
    memory_mb: Optional[float] = None
    inference_time_ms: Optional[float] = None
    gpu_utilization_pct: Optional[float] = None

    # Business metrics
    throughput_tps: Optional[float] = None  # transactions per second
    queue_wait_time_ms: Optional[float] = None
    error_rate_pct: Optional[float] = None

    metadata: Dict[str, Any] = field(default_factory=dict)


class ParetoDatabaseManager:
    """Manages database operations for Pareto analysis"""

    def __init__(self, db_url: str, pool_size: int = 10, max_overflow: int = 20):
        if not SQLALCHEMY_AVAILABLE:
            raise ImportError("SQLAlchemy required for database manager")

        self.engine = create_engine(
            db_url,
            poolclass=QueuePool,
            pool_size=pool_size,
            max_overflow=max_overflow,
            pool_pre_ping=True,
            echo=False
        )
        self.SessionLocal = sessionmaker(bind=self.engine)
        self.Base = declarative_base()
        self._create_tables()

    def _create_tables(self):
        """Create necessary tables if not exist"""

        class ParetoPointTable(self.Base):
            __tablename__ = 'pareto_points'

            id = Column(Integer, primary_key=True, autoincrement=True)
            tau_front = Column(Float, nullable=False)
            tau_back = Column(Float, nullable=False)
            cost = Column(Float, nullable=False)
            accuracy = Column(Float, nullable=False)
            f1_score = Column(Float)
            precision = Column(Float)
            recall = Column(Float)
            logs_processed_pct = Column(Float)

            timestamp = Column(DateTime, nullable=False)
            experiment_id = Column(String(255))
            run_id = Column(String(255))
            model_version = Column(String(255))
            environment = Column(String(50))

            cpu_usage_ms = Column(Float)
            memory_mb = Column(Float)
            inference_time_ms = Column(Float)
            gpu_utilization_pct = Column(Float)
            throughput_tps = Column(Float)
            queue_wait_time_ms = Column(Float)
            error_rate_pct = Column(Float)

            metadata_json = Column(JSON)

            __table_args__ = (
                Index('idx_tau_front_back', 'tau_front', 'tau_back'),
                Index('idx_timestamp', 'timestamp'),
                Index('idx_env_cost', 'environment', 'cost'),
                Index('idx_experiment', 'experiment_id', 'run_id'),
            )

        class ParetoFrontierTable(self.Base):
            __tablename__ = 'pareto_frontiers'

            id = Column(Integer, primary_key=True, autoincrement=True)
            frontier_version = Column(String(255), nullable=False)
            computed_at = Column(DateTime, nullable=False)
            num_points = Column(Integer)
            aupc = Column(Float)
            max_accuracy = Column(Float)
            min_cost = Column(Float)
            frontier_points_json = Column(JSON)
            parameters_json = Column(JSON)
            environment = Column(String(50))

        self.ParetoPointTable = ParetoPointTable
        self.ParetoFrontierTable = ParetoFrontierTable

        self.Base.metadata.create_all(self.engine)

    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """Get database session with context manager"""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def save_point(self, point: ParetoPoint) -> int:
        """Save a Pareto point to database"""
        with self.get_session() as session:
            db_point = self.ParetoPointTable(
                tau_front=point.tau_front,
                tau_back=point.tau_back,
                cost=point.cost,
                accuracy=point.accuracy,
                f1_score=point.f1_score,
                precision=point.precision,
                recall=point.recall,
                logs_processed_pct=point.logs_processed_pct,
                timestamp=point.timestamp,
                experiment_id=point.experiment_id,
                run_id=point.run_id,
                model_version=point.model_version,
                environment=point.environment,
                cpu_usage_ms=point.cpu_usage_ms,
                memory_mb=point.memory_mb,
                inference_time_ms=point.inference_time_ms,
                gpu_utilization_pct=point.gpu_utilization_pct,
                throughput_tps=point.throughput_tps,
                queue_wait_time_ms=point.queue_wait_time_ms,
                error_rate_pct=point.error_rate_pct,
                metadata_json=json.dumps(point.metadata) if point.metadata else None
            )
            session.add(db_point)
            session.flush()
            return db_point.id

    def save_frontier(self, frontier_version: str, frontier_points: List[ParetoPoint],
                      aupc: float, parameters: Dict[str, Any], environment: str = "production"):
        """Save Pareto frontier to database"""
        with self.get_session() as session:
            frontier_data = self.ParetoFrontierTable(
                frontier_version=frontier_version,
                computed_at=datetime.now(),
                num_points=len(frontier_points),
                aupc=aupc,
                max_accuracy=max(p.accuracy for p in frontier_points) if frontier_points else 0,
                min_cost=min(p.cost for p in frontier_points) if frontier_points else 0,
                frontier_points_json=json.dumps([asdict(p) for p in frontier_points]),
                parameters_json=json.dumps(parameters),
                environment=environment
            )
            session.add(frontier_data)

    def load_recent_points(self, hours: int = 24, environment: Optional[str] = None,
                           limit: int = 10000) -> List[ParetoPoint]:
        """Load recent Pareto points from database"""
        with self.get_session() as session:
            query = session.query(self.ParetoPointTable).filter(
                self.ParetoPointTable.timestamp >= datetime.now() - timedelta(hours=hours)
            )
            if environment:
                query = query.filter(self.ParetoPointTable.environment == environment)

            query = query.order_by(self.ParetoPointTable.timestamp.desc()).limit(limit)

            points = []
            for row in query.all():
                point = ParetoPoint(
                    tau_front=row.tau_front,
                    tau_back=row.tau_back,
                    cost=row.cost,
                    accuracy=row.accuracy,
                    f1_score=row.f1_score or 0,
                    precision=row.precision or 0,
                    recall=row.recall or 0,
                    logs_processed_pct=row.logs_processed_pct or 0,
                    timestamp=row.timestamp,
                    experiment_id=row.experiment_id,
                    run_id=row.run_id,
                    model_version=row.model_version,
                    environment=row.environment,
                    cpu_usage_ms=row.cpu_usage_ms,
                    memory_mb=row.memory_mb,
                    inference_time_ms=row.inference_time_ms,
                    gpu_utilization_pct=row.gpu_utilization_pct,
                    throughput_tps=row.throughput_tps,
                    queue_wait_time_ms=row.queue_wait_time_ms,
                    error_rate_pct=row.error_rate_pct,
                    metadata=json.loads(row.metadata_json) if row.metadata_json else {}
                )
                points.append(point)

            return points

    def get_latest_frontier(self, environment: str = "production") -> Optional[Dict]:
        """Get the latest computed Pareto frontier"""
        with self.get_session() as session:
            frontier = session.query(self.ParetoFrontierTable).filter(
                self.ParetoFrontierTable.environment == environment
            ).order_by(self.ParetoFrontierTable.computed_at.desc()).first()

            if frontier:
                return {
                    'frontier_version': frontier.frontier_version,
                    'computed_at': frontier.computed_at,
                    'num_points': frontier.num_points,
                    'aupc': frontier.aupc,
                    'max_accuracy': frontier.max_accuracy,
                    'min_cost': frontier.min_cost,
                    'frontier_points': json.loads(frontier.frontier_points_json),
                    'parameters': json.loads(frontier.parameters_json)
                }
            return None


class ParetoMetricsCollector:
    """Prometheus metrics collector for Pareto analyzer"""

    def __init__(self, namespace: str = "caaapt"):
        self.points_total = Counter(
            f'{namespace}_pareto_points_total',
            'Total Pareto points collected',
            ['environment', 'operation_mode']
        )
        self.frontier_computations = Counter(
            f'{namespace}_frontier_computations_total',
            'Total frontier computations',
            ['environment', 'status']
        )
        self.computation_time = Histogram(
            f'{namespace}_pareto_computation_seconds',
            'Time spent computing Pareto frontier',
            buckets=[0.1, 0.5, 1, 2, 5, 10]
        )
        self.current_frontier_points = Gauge(
            f'{namespace}_current_frontier_points',
            'Number of points on current Pareto frontier'
        )
        self.best_accuracy = Gauge(
            f'{namespace}_best_accuracy',
            'Best accuracy on current frontier'
        )
        self.lowest_cost = Gauge(
            f'{namespace}_lowest_cost',
            'Lowest cost on current frontier'
        )


class ProductionParetoAnalyzer:
    """
    Production-grade Pareto analyzer with database integration, streaming support,
    and ML pipeline integration.
    """

    def __init__(
            self,
            db_url: Optional[str] = None,
            redis_url: Optional[str] = None,
            kafka_bootstrap_servers: Optional[str] = None,
            mlflow_tracking_uri: Optional[str] = None,
            mode: ParetoOperationMode = ParetoOperationMode.HYBRID,
            cost_strategy: CostStrategy = CostStrategy.HYBRID_COST,
            environment: str = "production",
            update_interval_seconds: int = 60,
            enable_auto_compute: bool = True
    ):
        """
        Initialize production Pareto analyzer

        Args:
            db_url: PostgreSQL/MySQL URL for persistent storage
            redis_url: Redis URL for caching
            kafka_bootstrap_servers: Kafka servers for streaming
            mlflow_tracking_uri: MLflow tracking URI
            mode: Operation mode (batch/streaming/hybrid)
            cost_strategy: Cost calculation strategy
            environment: Deployment environment
            update_interval_seconds: Auto-update interval for streaming mode
            enable_auto_compute: Automatically recompute frontier
        """
        self.mode = mode
        self.cost_strategy = cost_strategy
        self.environment = environment
        self.update_interval = update_interval_seconds
        self.enable_auto_compute = enable_auto_compute

        # Initialize components
        self.db_manager = ParetoDatabaseManager(db_url) if db_url else None
        self.metrics = ParetoMetricsCollector()

        # Redis for caching
        self.redis_client = None
        if REDIS_AVAILABLE and redis_url:
            self.redis_client = redis.Redis.from_url(redis_url, decode_responses=True)

        # Kafka for streaming
        self.kafka_producer = None
        self.kafka_consumer = None
        if KAFKA_AVAILABLE and kafka_bootstrap_servers:
            self.kafka_producer = KafkaProducer(
                bootstrap_servers=kafka_bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode('utf-8')
            )

        # MLflow integration
        if mlflow_tracking_uri:
            mlflow.set_tracking_uri(mlflow_tracking_uri)
        self.mlflow_client = MlflowClient() if mlflow_tracking_uri else None

        # In-memory storage
        self.points: List[ParetoPoint] = []
        self.frontier_points: List[ParetoPoint] = []
        self.baseline_points: Dict[str, ParetoPoint] = {}
        self.full_llm_cost_baseline = 1.0

        # Threading
        self._stop_event = threading.Event()
        self._compute_thread = None
        self._streaming_thread = None

        # Start background tasks if needed
        if self.mode in [ParetoOperationMode.STREAMING, ParetoOperationMode.HYBRID]:
            self._start_background_tasks()

        # Load historical data if available
        self._load_historical_data()

    def _start_background_tasks(self):
        """Start background threads for streaming and auto-computation"""
        if self.enable_auto_compute:
            self._compute_thread = threading.Thread(target=self._auto_compute_loop, daemon=True)
            self._compute_thread.start()

        if self.mode in [ParetoOperationMode.STREAMING, ParetoOperationMode.HYBRID]:
            self._streaming_thread = threading.Thread(target=self._streaming_loop, daemon=True)
            self._streaming_thread.start()

    def _auto_compute_loop(self):
        """Auto-compute Pareto frontier periodically"""
        while not self._stop_event.is_set():
            self._stop_event.wait(self.update_interval)
            try:
                self.compute_pareto_frontier_and_save()
                logger.info(f"Auto-computed Pareto frontier at {datetime.now()}")
            except Exception as e:
                logger.error(f"Auto-compute failed: {e}")

    def _streaming_loop(self):
        """Process streaming Pareto points from Kafka"""
        if not self.kafka_producer:
            return

        # This would connect to Kafka topic; for demo we'll simulate
        while not self._stop_event.is_set():
            self._stop_event.wait(1)
            # Actual Kafka consumption would go here
            pass

    def _load_historical_data(self):
        """Load historical data from database"""
        if self.db_manager:
            try:
                points = self.db_manager.load_recent_points(
                    hours=168,  # 7 days
                    environment=self.environment,
                    limit=50000
                )
                self.points.extend(points)
                logger.info(f"Loaded {len(points)} historical Pareto points")
            except Exception as e:
                logger.error(f"Failed to load historical data: {e}")

    def add_point(self, point: ParetoPoint):
        """Add a configuration point to the analysis"""
        point.environment = self.environment
        self.points.append(point)

        # Save to database
        if self.db_manager:
            try:
                point_id = self.db_manager.save_point(point)
                logger.debug(f"Saved Pareto point {point_id}")
            except Exception as e:
                logger.error(f"Failed to save point to DB: {e}")

        # Cache in Redis
        if self.redis_client:
            try:
                key = f"pareto:point:{point.timestamp.timestamp()}"
                self.redis_client.setex(key, 3600, json.dumps(asdict(point)))
            except Exception as e:
                logger.error(f"Failed to cache point: {e}")

        # Send to Kafka for downstream processing
        if self.kafka_producer:
            try:
                self.kafka_producer.send('pareto-points', asdict(point))
            except Exception as e:
                logger.error(f"Failed to send to Kafka: {e}")

        # Update metrics
        self.metrics.points_total.labels(environment=self.environment, operation_mode=self.mode.value).inc()

    def add_points_from_mlflow_run(self, run_id: str):
        """Load points from MLflow run"""
        if not self.mlflow_client:
            logger.warning("MLflow client not initialized")
            return

        try:
            run = self.mlflow_client.get_run(run_id)
            metrics = run.data.metrics

            point = ParetoPoint(
                tau_front=float(run.data.params.get('tau_front', 0)),
                tau_back=float(run.data.params.get('tau_back', 0)),
                cost=metrics.get('cost', 0),
                accuracy=metrics.get('tactic_accuracy', 0),
                f1_score=metrics.get('f1_score', 0),
                precision=metrics.get('precision', 0),
                recall=metrics.get('recall', 0),
                logs_processed_pct=metrics.get('logs_processed_pct', 0),
                run_id=run_id,
                experiment_id=run.info.experiment_id,
                model_version=run.data.params.get('model_version')
            )
            self.add_point(point)
            logger.info(f"Added point from MLflow run {run_id}")
        except Exception as e:
            logger.error(f"Failed to load from MLflow: {e}")

    def add_batch_points(self, points: List[ParetoPoint]):
        """Add multiple points in batch"""
        for point in points:
            self.add_point(point)

        # Trigger recompute if many points added
        if len(points) > 100 and self.enable_auto_compute:
            self.compute_pareto_frontier_and_save()

    def compute_pareto_frontier(self, maximize_accuracy: bool = True) -> List[ParetoPoint]:
        """Compute the Pareto frontier from all points"""
        with self.metrics.computation_time.time():
            if not self.points:
                return []

            # Convert points to list of (cost, accuracy)
            points_list = [(p.cost, p.accuracy, p) for p in self.points]
            points_list.sort(key=lambda x: (x[0], -x[1] if maximize_accuracy else x[1]))

            # Compute Pareto frontier
            frontier = []
            for cost, accuracy, point in points_list:
                is_dominated = False

                if maximize_accuracy:
                    for existing_cost, existing_accuracy, _ in frontier:
                        if existing_cost <= cost and existing_accuracy >= accuracy:
                            is_dominated = True
                            break

                    if not is_dominated:
                        frontier = [(c, a, p) for c, a, p in frontier
                                    if not (c >= cost and a <= accuracy)]
                        frontier.append((cost, accuracy, point))
                        frontier.sort(key=lambda x: x[0])
                else:
                    # For other scenarios
                    pass

            self.frontier_points = [p for _, _, p in frontier]

            # Update Prometheus metrics
            self.metrics.current_frontier_points.set(len(self.frontier_points))
            if self.frontier_points:
                self.metrics.best_accuracy.set(max(p.accuracy for p in self.frontier_points))
                self.metrics.lowest_cost.set(min(p.cost for p in self.frontier_points))

            logger.info(f"Computed Pareto frontier with {len(self.frontier_points)} points")
            return self.frontier_points

    def compute_pareto_frontier_and_save(self, frontier_version: Optional[str] = None):
        """Compute frontier and save to database"""
        frontier = self.compute_pareto_frontier()

        if self.db_manager and frontier_version:
            aupc = self.compute_auc()
            self.db_manager.save_frontier(
                frontier_version=frontier_version,
                frontier_points=frontier,
                aupc=aupc,
                parameters={
                    'mode': self.mode.value,
                    'cost_strategy': self.cost_strategy.value,
                    'num_points': len(self.points)
                },
                environment=self.environment
            )

            # Log to MLflow
            if self.mlflow_client:
                with mlflow.start_run(run_name=f"pareto_frontier_{frontier_version}"):
                    mlflow.log_metric("aupc", aupc)
                    mlflow.log_metric("frontier_points", len(frontier))
                    mlflow.log_param("environment", self.environment)

    def optimize_for_deployment(self, constraints: Dict[str, Any]) -> ParetoPoint:
        """
        Find optimal configuration given deployment constraints

        Args:
            constraints: dict with keys like 'max_cost', 'min_accuracy',
                        'preference' ('cost' or 'accuracy')

        Returns:
            Optimal Pareto point
        """
        if not self.frontier_points:
            self.compute_pareto_frontier()

        if not self.frontier_points:
            raise ValueError("No frontier points available")

        max_cost = constraints.get('max_cost', float('inf'))
        min_accuracy = constraints.get('min_accuracy', 0)
        preference = constraints.get('preference', 'balanced')

        # Filter by constraints
        valid_points = [p for p in self.frontier_points
                        if p.cost <= max_cost and p.accuracy >= min_accuracy]

        if not valid_points:
            logger.warning("No points satisfy constraints, returning closest")
            valid_points = self.frontier_points

        if preference == 'cost':
            return min(valid_points, key=lambda x: x.cost)
        elif preference == 'accuracy':
            return max(valid_points, key=lambda x: x.accuracy)
        else:  # balanced
            # Find point closest to ideal (min cost, max accuracy)
            min_cost = min(p.cost for p in valid_points)
            max_accuracy = max(p.accuracy for p in valid_points)

            def distance(point):
                cost_norm = (point.cost - min_cost) / (max_cost - min_cost + 1e-6)
                acc_norm = (max_accuracy - point.accuracy) / (max_accuracy - min_cost + 1e-6)
                return cost_norm + acc_norm

            return min(valid_points, key=distance)

    def get_live_recommendations(self, current_throughput: float,
                                 latency_budget_ms: float) -> Dict[str, Any]:
        """
        Get real-time recommendations based on current system state

        Args:
            current_throughput: Current TPS
            latency_budget_ms: Maximum allowed latency

        Returns:
            Recommended tau_front, tau_back values
        """
        if not self.frontier_points:
            self.compute_pareto_frontier()

        # Estimate cost based on throughput
        target_cost = min(1.0, (latency_budget_ms / 1000.0) / current_throughput)

        # Find point with closest cost
        closest = min(self.frontier_points, key=lambda x: abs(x.cost - target_cost))

        return {
            'recommended_tau_front': closest.tau_front,
            'recommended_tau_back': closest.tau_back,
            'expected_cost': closest.cost,
            'expected_accuracy': closest.accuracy,
            'expected_throughput': closest.throughput_tps or current_throughput,
            'confidence': 1.0 - abs(closest.cost - target_cost) / max(target_cost, 0.01)
        }

    def compute_auc(self) -> float:
        """Compute Area Under Pareto Curve"""
        if len(self.frontier_points) < 2:
            return 0.0

        sorted_points = sorted(self.frontier_points, key=lambda x: x.cost)
        costs = [p.cost for p in sorted_points]
        accuracies = [p.accuracy for p in sorted_points]

        if costs[0] > 0:
            costs.insert(0, 0)
            accuracies.insert(0, accuracies[0])
        if costs[-1] < 1:
            costs.append(1)
            accuracies.append(accuracies[-1])

        return auc(costs, accuracies)

    def generate_deployment_guidelines(self) -> Dict[str, Dict[str, Any]]:
        """Generate deployment guidelines based on live data"""
        if not self.frontier_points:
            self.compute_pareto_frontier()

        regions = self._get_characteristic_regions()
        guidelines = {}

        for region_name, points in regions.items():
            if points:
                # Use median for robustness
                median_point = sorted(points, key=lambda x: x.accuracy)[len(points) // 2]
                guidelines[region_name] = {
                    'description': self._get_region_description(region_name),
                    'tau_front': median_point.tau_front,
                    'tau_back': median_point.tau_back,
                    'expected_cost': median_point.cost,
                    'expected_accuracy': median_point.accuracy,
                    'expected_latency_ms': median_point.inference_time_ms,
                    'expected_throughput_tps': median_point.throughput_tps,
                    'use_case': self._get_use_case(region_name)
                }
            else:
                guidelines[region_name] = self._get_default_guideline(region_name)

        return guidelines

    def _get_characteristic_regions(self) -> Dict[str, List[ParetoPoint]]:
        """Classify frontier into regions"""
        regions = {'ultra_low_cost': [], 'balanced': [], 'high_precision': []}

        for point in self.frontier_points:
            if point.cost < 0.01:
                regions['ultra_low_cost'].append(point)
            elif 0.01 <= point.cost < 0.10:
                regions['balanced'].append(point)
            else:
                regions['high_precision'].append(point)

        return regions

    def _get_region_description(self, region: str) -> str:
        """Get description for region"""
        descriptions = {
            'ultra_low_cost': 'Maximum cost efficiency for high-volume processing',
            'balanced': 'Optimal trade-off for routine operations',
            'high_precision': 'Maximum accuracy for forensic analysis'
        }
        return descriptions.get(region, 'General purpose configuration')

    def _get_use_case(self, region: str) -> str:
        """Get use case for region"""
        use_cases = {
            'ultra_low_cost': 'Real-time SIEM, high-volume log processing',
            'balanced': 'Security operations center (SOC), daily analysis',
            'high_precision': 'Post-incident forensics, advanced threat hunting'
        }
        return use_cases.get(region, 'General security analysis')

    def _get_default_guideline(self, region: str) -> Dict:
        """Get default guideline when no data available"""
        defaults = {
            'ultra_low_cost': {
                'tau_front': 0.95, 'tau_back': 0.75, 'expected_cost': 0.003,
                'expected_accuracy': 95.5, 'use_case': 'Large-scale log auditing'
            },
            'balanced': {
                'tau_front': 0.70, 'tau_back': 0.85, 'expected_cost': 0.018,
                'expected_accuracy': 95.6, 'use_case': 'Routine security operations'
            },
            'high_precision': {
                'tau_front': 0.30, 'tau_back': 0.90, 'expected_cost': 0.124,
                'expected_accuracy': 94.2, 'use_case': 'Post-incident forensics'
            }
        }
        guideline = defaults.get(region, defaults['balanced'])
        guideline['description'] = self._get_region_description(region)
        return guideline

    def stop(self):
        """Gracefully stop background tasks"""
        self._stop_event.set()
        if self._compute_thread:
            self._compute_thread.join(timeout=5)
        if self._streaming_thread:
            self._streaming_thread.join(timeout=5)
        if self.kafka_producer:
            self.kafka_producer.close()

    def get_summary_statistics(self) -> Dict[str, Any]:
        """Get comprehensive summary statistics"""
        if not self.frontier_points:
            self.compute_pareto_frontier()

        if not self.frontier_points:
            return {'error': 'No frontier points available'}

        return {
            'num_points_total': len(self.points),
            'num_frontier_points': len(self.frontier_points),
            'max_accuracy_pct': max(p.accuracy for p in self.frontier_points),
            'min_normalized_cost': min(p.cost for p in self.frontier_points),
            'aupc': self.compute_auc(),
            'last_updated': datetime.now().isoformat(),
            'operation_mode': self.mode.value,
            'environment': self.environment,
            'characteristic_regions': {
                region: len(points)
                for region, points in self._get_characteristic_regions().items()
            }
        }


# Integration with existing ML pipeline
class ParetoMLflowIntegration:
    """Integrate Pareto analyzer with MLflow pipeline"""

    def __init__(self, analyzer: ProductionParetoAnalyzer):
        self.analyzer = analyzer

    def log_pareto_metrics(self, run_id: str):
        """Log Pareto metrics to current MLflow run"""
        stats = self.analyzer.get_summary_statistics()

        for key, value in stats.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(f"pareto_{key}", value)

        # Log optimal deployment configs
        guidelines = self.analyzer.generate_deployment_guidelines()
        for region, config in guidelines.items():
            mlflow.log_params({
                f"pareto_{region}_tau_front": config['tau_front'],
                f"pareto_{region}_tau_back": config['tau_back'],
                f"pareto_{region}_cost": config['expected_cost']
            })

    def register_best_model(self, model_name: str, stage: str = "Production"):
        """Register the best model based on Pareto analysis"""
        guidelines = self.analyzer.generate_deployment_guidelines()

        # Choose balanced configuration for production
        balanced = guidelines.get('balanced', {})

        # Register model with staging/production
        # Implementation depends on your model registry


# Helper function for quick deployment
def create_production_analyzer(
        db_url: str,
        redis_url: Optional[str] = None,
        environment: str = "production"
) -> ProductionParetoAnalyzer:
    """
    Quick factory function to create a production-ready Pareto analyzer

    Args:
        db_url: Database URL (PostgreSQL recommended)
        redis_url: Redis URL for caching
        environment: Deployment environment

    Returns:
        Configured ProductionParetoAnalyzer
    """
    return ProductionParetoAnalyzer(
        db_url=db_url,
        redis_url=redis_url,
        environment=environment,
        mode=ParetoOperationMode.HYBRID,
        cost_strategy=CostStrategy.HYBRID_COST,
        update_interval_seconds=60,
        enable_auto_compute=True
    )