#!/usr/bin/env python3
"""
TREC Dataset Preprocessor for CAAAPT Framework

This script processes the TREC APT Tactic Dataset for provenance graph-based
APT tactic/technique recognition. Compatible with CAAAPT's feature extraction pipeline.

Dataset source: https://www.kellect.org/#/kellect-4-aptdataset
Paper: "TREC: APT Tactic/Technique Recognition via Few-Shot Provenance Subgraph Learning" (CCS 2024)
"""

import os
import json
import pickle
import argparse
import hashlib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Set
from collections import defaultdict, Counter
from dataclasses import dataclass, field, asdict
from enum import Enum
import warnings

warnings.filterwarnings('ignore')

# Optional imports with fallbacks
try:
    import networkx as nx

    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    print("Warning: networkx not installed. Graph features disabled.")

try:
    from tqdm import tqdm

    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


    # Fallback tqdm
    def tqdm(iterable, *args, **kwargs):
        return iterable


class EntityType(Enum):
    """Entity types from paper Table 1"""
    PROCESS = "Process"
    FILE = "File"
    SOCKET = "Socket"


class EdgeType(Enum):
    """Edge types from paper Table 1"""
    WRITE = "Write"
    READ = "Read"
    EXECUTE = "Execute"
    FORK = "Fork"
    SEND = "Send"
    RECEIVE = "Receive"
    MMAP = "Mmap"


@dataclass
class EventRecord:
    """
    System event record matching paper Table 1 structure

    Attributes:
        subject_type: Type of source entity (Process, File, Socket)
        subject_id: Unique identifier for source entity
        object_type: Type of target entity (Process, File, Socket)
        object_id: Unique identifier for target entity
        edge_type: Type of interaction (Write, Read, Execute, Fork, Send, Receive, Mmap)
        timestamp: Event timestamp (integer)
        attributes: Additional entity attributes (pathname, IP addresses, UUID, etc.)
    """
    subject_type: str
    subject_id: str
    object_type: str
    object_id: str
    edge_type: str
    timestamp: int
    attributes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> 'EventRecord':
        return cls(**data)


@dataclass
class CompressedEdge:
    """
    Compressed edge structure from paper Section 3.1.1

    Semantically equivalent events are compressed with frequency counts.
    """
    event: EventRecord
    frequency: int
    first_seen: int
    last_seen: int


@dataclass
class AttackSample:
    """
    Complete attack sample with provenance graph and labels

    Matches paper's data structure with benign/malicious node counts.
    Average sample has 8,957 nodes with only 4 malicious nodes (<0.05%).
    """
    sample_id: str
    events: List[EventRecord]
    graph: Any  # nx.DiGraph if networkx available
    tactic_label: str
    technique_label: Optional[str] = None
    benign_nodes_count: int = 0
    malicious_nodes_count: int = 0
    total_edges: int = 0
    compressed_events: List[CompressedEdge] = field(default_factory=list)

    def to_dict(self) -> Dict:
        result = {
            'sample_id': self.sample_id,
            'tactic_label': self.tactic_label,
            'technique_label': self.technique_label,
            'benign_nodes_count': self.benign_nodes_count,
            'malicious_nodes_count': self.malicious_nodes_count,
            'total_edges': self.total_edges,
            'events': [e.to_dict() for e in self.events]
        }
        if self.compressed_events:
            result['compressed_events'] = [
                {'frequency': ce.frequency, 'first_seen': ce.first_seen, 'last_seen': ce.last_seen,
                 'event': ce.event.to_dict()}
                for ce in self.compressed_events
            ]
        return result


class SemanticCompressor:
    """
    Semantic-preserving data compressor from paper Section 3.1.1

    Retains only first event when source semantics unchanged, records frequency.
    """

    # Event types that change source semantics
    SEMANTIC_CHANGING_EVENTS = {'Read', 'Execute', 'Receive'}

    # Event types that can be compressed (only track frequency)
    COMPRESSIBLE_EVENTS = {'Write', 'Mmap', 'Send'}

    def __init__(self):
        self._source_semantics: Dict[str, Dict[str, Any]] = {}

    def get_semantic_key(self, event: EventRecord) -> str:
        """
        Generate semantic key for source entity.
        Semantics determined by entity type and its mutable attributes.
        """
        if event.subject_type == EntityType.PROCESS.value:
            # Process semantics: pathname + UUID (from paper Table 1)
            attrs = event.attributes
            return f"proc:{attrs.get('pathname', '')}:{attrs.get('Uuid', '')}"
        elif event.subject_type == EntityType.FILE.value:
            # File semantics: pathname (content hash not available in logs)
            return f"file:{event.attributes.get('pathname', '')}"
        elif event.subject_type == EntityType.SOCKET.value:
            # Socket semantics: source IP + destination IP + UUID
            attrs = event.attributes
            return f"socket:{attrs.get('source_ip', '')}:{attrs.get('dest_ip', '')}:{attrs.get('Uuid', '')}"
        return f"{event.subject_type}:{event.subject_id}"

    def compress_events(self, events: List[EventRecord]) -> List[CompressedEdge]:
        """
        Apply semantic-preserving compression from paper Figure 2.

        Example: Three Write events from F1 at t=2,3,4 are semantically equivalent,
        so only first edge retained with frequency=3.
        """
        compressed: List[CompressedEdge] = []

        # Group events by (subject, object, edge_type) for compression candidates
        compression_groups: Dict[Tuple, List[EventRecord]] = defaultdict(list)

        for event in events:
            if event.edge_type in self.COMPRESSIBLE_EVENTS:
                # Compressible: group by semantic key
                key = (
                    self.get_semantic_key(event),
                    event.object_type,
                    event.object_id,
                    event.edge_type
                )
                compression_groups[key].append(event)
            else:
                # Non-compressible: keep as-is (semantic-changing events)
                compressed.append(CompressedEdge(
                    event=event,
                    frequency=1,
                    first_seen=event.timestamp,
                    last_seen=event.timestamp
                ))

        # Process compressible groups
        for key, group_events in compression_groups.items():
            if not group_events:
                continue

            # Sort by timestamp
            sorted_events = sorted(group_events, key=lambda e: e.timestamp)

            # Keep first event, record frequency
            first_event = sorted_events[0]
            compressed.append(CompressedEdge(
                event=first_event,
                frequency=len(sorted_events),
                first_seen=first_event.timestamp,
                last_seen=sorted_events[-1].timestamp
            ))

        # Sort by timestamp
        compressed.sort(key=lambda ce: ce.first_seen)

        return compressed


class TRECDatasetProcessor:
    """
    Main processor for TREC APT Tactic Dataset

    Based on: https://www.kellect.org/#/kellect-4-aptdataset
    """

    def __init__(self, data_root: str, output_dir: str):
        self.data_root = Path(data_root)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.compressor = SemanticCompressor()
        self.samples: List[AttackSample] = []

        # Statistics tracking
        self.stats = {
            'total_samples': 0,
            'tactic_distribution': Counter(),
            'avg_events_per_sample': 0,
            'avg_malicious_nodes': 0,
            'total_benign_nodes': 0,
            'total_malicious_nodes': 0
        }

    def _generate_entity_id(self, entity_type: str, attributes: Dict) -> str:
        """Generate unique entity ID from attributes"""
        if entity_type == 'process':
            key = f"proc:{attributes.get('pid', '')}:{attributes.get('pathname', '')}"
        elif entity_type == 'file':
            key = f"file:{attributes.get('pathname', '')}"
        elif entity_type == 'socket':
            key = f"socket:{attributes.get('source_ip', '')}:{attributes.get('dest_ip', '')}"
        else:
            key = f"{entity_type}:{str(attributes)}"
        return hashlib.md5(key.encode()).hexdigest()[:16]

    def _parse_event(self, raw_event: Dict, sample_id: str) -> Optional[EventRecord]:
        """Parse raw event JSON to EventRecord"""
        try:
            # Extract event type
            edge_type_raw = raw_event.get('type', raw_event.get('edge_type', 'unknown')).lower()
            edge_type = EdgeTypeMapping.get(edge_type_raw, 'Read')

            # Extract entity types
            subject_type_raw = raw_event.get('subject_type', raw_event.get('src_type', 'process'))
            object_type_raw = raw_event.get('object_type', raw_event.get('dst_type', 'file'))

            subject_type = EntityTypeMapping.get(subject_type_raw.lower(), 'Process')
            object_type = EntityTypeMapping.get(object_type_raw.lower(), 'File')

            # Extract IDs
            subject_id = raw_event.get('subject_id', raw_event.get('src_id',
                                                                   self._generate_entity_id(subject_type, raw_event.get(
                                                                       'subject_attrs', {}))))
            object_id = raw_event.get('object_id', raw_event.get('dst_id',
                                                                 self._generate_entity_id(object_type,
                                                                                          raw_event.get('object_attrs',
                                                                                                        {}))))

            # Extract timestamp
            timestamp = raw_event.get('timestamp', raw_event.get('time', 0))

            # Extract attributes
            attributes = {
                'pathname': raw_event.get('pathname', ''),
                'Uuid': raw_event.get('uuid', raw_event.get('Uuid', sample_id)),
                'source_ip': raw_event.get('src_ip', ''),
                'destination_ip': raw_event.get('dst_ip', '')
            }
            # Merge additional attributes
            for key in ['pid', 'ppid', 'uid', 'gid', 'size', 'mode']:
                if key in raw_event:
                    attributes[key] = raw_event[key]

            return EventRecord(
                subject_type=subject_type,
                subject_id=str(subject_id),
                object_type=object_type,
                object_id=str(object_id),
                edge_type=edge_type,
                timestamp=int(timestamp),
                attributes=attributes
            )
        except Exception as e:
            print(f"Warning: Failed to parse event in {sample_id}: {e}")
            return None

    def _build_graph(self, events: List[EventRecord]):
        """Build provenance graph from events"""
        if not HAS_NETWORKX:
            return None

        G = nx.DiGraph()

        for event in events:
            # Add nodes
            src_node = f"{event.subject_type}:{event.subject_id}"
            dst_node = f"{event.object_type}:{event.object_id}"

            G.add_node(src_node, type=event.subject_type, attrs=event.attributes)
            G.add_node(dst_node, type=event.object_type)

            # Add edge
            G.add_edge(src_node, dst_node,
                       type=event.edge_type,
                       timestamp=event.timestamp)

        return G

    def load_raw_data(self, raw_data_path: str) -> List[Dict]:
        """
        Load raw TREC dataset.

        Expected format (based on Kellect dataset):
        - JSON lines format or JSON array
        - Each sample: {sample_id, tactic, technique, events, ground_truth_nodes}
        """
        raw_path = Path(raw_data_path)
        samples = []

        if raw_path.is_dir():
            # Load all JSON files in directory
            json_files = list(raw_path.glob('*.json'))
            for json_file in tqdm(json_files, desc="Loading files"):
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        samples.extend(data)
                    else:
                        samples.append(data)
        else:
            # Single file
            with open(raw_path, 'r') as f:
                if raw_path.suffix == '.jsonl':
                    for line in f:
                        if line.strip():
                            samples.append(json.loads(line))
                else:
                    data = json.load(f)
                    samples = data if isinstance(data, list) else [data]

        return samples

    def process_sample(self, raw_sample: Dict) -> Optional[AttackSample]:
        """Process single sample from raw data"""
        sample_id = raw_sample.get('sample_id', raw_sample.get('id', 'unknown'))
        tactic = raw_sample.get('tactic', raw_sample.get('tactic_label', 'unknown'))
        technique = raw_sample.get('technique', raw_sample.get('technique_label', None))

        # Parse events
        raw_events = raw_sample.get('events', raw_sample.get('provenance_graph', []))
        events = []
        for raw_event in raw_events:
            event = self._parse_event(raw_event, sample_id)
            if event:
                events.append(event)

        if not events:
            print(f"Warning: No valid events for sample {sample_id}")
            return None

        # Apply semantic compression (paper Section 3.1.1)
        compressed_events = self.compressor.compress_events(events)

        # Build graph
        graph = self._build_graph(events)

        # Count malicious vs benign (from ground truth or high-confidence labels)
        malicious_nodes = raw_sample.get('malicious_nodes', raw_sample.get('attack_nodes', []))
        malicious_count = len(malicious_nodes) if malicious_nodes else 0
        benign_count = len(set([e.subject_id for e in events])) - malicious_count

        # Update statistics
        self.stats['tactic_distribution'][tactic] += 1

        return AttackSample(
            sample_id=sample_id,
            events=events,
            graph=graph,
            tactic_label=tactic,
            technique_label=technique,
            benign_nodes_count=benign_count,
            malicious_nodes_count=malicious_count,
            total_edges=len(events),
            compressed_events=compressed_events
        )

    def process_dataset(self, input_path: str, max_samples: int = None) -> List[AttackSample]:
        """Process entire dataset"""
        print(f"Loading data from {input_path}...")
        raw_samples = self.load_raw_data(input_path)

        if max_samples:
            raw_samples = raw_samples[:max_samples]

        print(f"Processing {len(raw_samples)} samples...")
        for raw_sample in tqdm(raw_samples, desc="Processing samples"):
            sample = self.process_sample(raw_sample)
            if sample:
                self.samples.append(sample)

        self._compute_statistics()
        return self.samples

    def _compute_statistics(self):
        """Compute dataset statistics"""
        self.stats['total_samples'] = len(self.samples)

        if self.samples:
            total_events = sum(s.total_edges for s in self.samples)
            self.stats['avg_events_per_sample'] = total_events / len(self.samples)
            self.stats['avg_malicious_nodes'] = np.mean([s.malicious_nodes_count for s in self.samples])
            self.stats['total_benign_nodes'] = sum(s.benign_nodes_count for s in self.samples)
            self.stats['total_malicious_nodes'] = sum(s.malicious_nodes_count for s in self.samples)

    def save_processed_data(self):
        """Save processed samples in multiple formats for CAAAPT"""

        # 1. Save as JSON (complete samples)
        json_path = self.output_dir / 'trec_samples.json'
        with open(json_path, 'w') as f:
            json.dump([s.to_dict() for s in self.samples], f, indent=2)
        print(f"Saved JSON to {json_path}")

        # 2. Save as pickle (faster loading for large datasets)
        pickle_path = self.output_dir / 'trec_samples.pkl'
        with open(pickle_path, 'wb') as f:
            pickle.dump(self.samples, f)
        print(f"Saved pickle to {pickle_path}")

        # 3. Save event sequences for frontend feature extraction
        sequences_path = self.output_dir / 'event_sequences.json'
        sequences = []
        for sample in self.samples:
            seq = {
                'sample_id': sample.sample_id,
                'tactic': sample.tactic_label,
                'technique': sample.technique_label,
                'event_sequence': [
                    f"{e.subject_type}|{e.edge_type}|{e.object_type}"
                    for e in sample.events
                ],
                'compressed_sequence': [
                    f"{ce.event.edge_type}:{ce.frequency}"
                    for ce in sample.compressed_events
                ],
                'malicious_ratio': sample.malicious_nodes_count / max(1,
                                                                      sample.benign_nodes_count + sample.malicious_nodes_count)
            }
            sequences.append(seq)

        with open(sequences_path, 'w') as f:
            json.dump(sequences, f, indent=2)
        print(f"Saved sequences to {sequences_path}")

        # 4. Save labels for training
        labels_df = pd.DataFrame([
            {'sample_id': s.sample_id, 'tactic': s.tactic_label, 'technique': s.technique_label}
            for s in self.samples
        ])
        labels_path = self.output_dir / 'labels.csv'
        labels_df.to_csv(labels_path, index=False)
        print(f"Saved labels to {labels_path}")

        # 5. Save statistics
        stats_path = self.output_dir / 'dataset_stats.json'
        with open(stats_path, 'w') as f:
            json.dump({
                'total_samples': self.stats['total_samples'],
                'tactic_distribution': dict(self.stats['tactic_distribution']),
                'avg_events_per_sample': self.stats['avg_events_per_sample'],
                'avg_malicious_nodes': self.stats['avg_malicious_nodes'],
                'total_benign_nodes': self.stats['total_benign_nodes'],
                'total_malicious_nodes': self.stats['total_malicious_nodes']
            }, f, indent=2)
        print(f"Saved statistics to {stats_path}")

    def print_summary(self):
        """Print dataset summary matching paper Table 3"""
        print("\n" + "=" * 60)
        print("TREC Dataset Summary (Paper Table 3)")
        print("=" * 60)
        print(f"Total samples: {self.stats['total_samples']}")
        print(f"Average events per sample: {self.stats['avg_events_per_sample']:.2f}")
        print(f"Average malicious nodes per sample: {self.stats['avg_malicious_nodes']:.2f}")
        print(f"Benign nodes total: {self.stats['total_benign_nodes']:,}")
        print(f"Malicious nodes total: {self.stats['total_malicious_nodes']:,}")
        print(
            f"Malicious ratio: {self.stats['total_malicious_nodes'] / max(1, self.stats['total_benign_nodes'] + self.stats['total_malicious_nodes']):.4%}")

        print("\nTactic Distribution:")
        print("-" * 40)
        for tactic, count in sorted(self.stats['tactic_distribution'].items(), key=lambda x: -x[1]):
            print(f"  {tactic:25s}: {count:4d} samples")


def create_synthetic_demo(output_dir: str):
    """
    Create synthetic demo data for testing when real dataset not available.

    This matches the distribution from paper Table 3.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Paper Table 3 distribution
    tactic_distribution = {
        'Initial Access': 44,
        'Lateral Movement': 50,
        'Credential Access': 34,
        'Defense Evasion': 78,
        'Persistence': 25,
        'Privilege Escalation': 30,
        'Discovery': 15,
        'Execution': 23,
        'Exfiltration': 5
    }

    samples = []
    sample_id = 1

    for tactic, count in tactic_distribution.items():
        for i in range(count):
            # Generate event sequence (average 8,957 nodes from paper)
            num_events = np.random.randint(50, 200)
            events = []

            # Malicious nodes are rare (<0.05% as per paper)
            malicious_node_ids = set(np.random.choice(range(1000), size=max(1, int(1000 * 0.0005)), replace=False))

            for j in range(num_events):
                # Alternate between benign and potentially malicious
                is_malicious = j in malicious_node_ids

                event = {
                    'type': np.random.choice(['Write', 'Read', 'Execute', 'Fork', 'Send', 'Receive', 'Mmap']),
                    'subject_type': np.random.choice(['process', 'file', 'socket']),
                    'object_type': np.random.choice(['process', 'file', 'socket']),
                    'timestamp': j * 1000,
                    'pid': np.random.randint(1000, 10000),
                    'pathname': f"/proc/{np.random.randint(1, 100)}/exe",
                    'Uuid': hashlib.md5(f"{sample_id}_{j}".encode()).hexdigest()[:8],
                    'is_malicious': is_malicious
                }
                events.append(event)

            # Ground truth malicious nodes
            ground_truth = [f"node_{nid}" for nid in malicious_node_ids]

            samples.append({
                'sample_id': f"TREC_{tactic.replace(' ', '_')}_{sample_id:04d}",
                'tactic': tactic,
                'technique': f"T{np.random.randint(1000, 2000)}",
                'events': events,
                'malicious_nodes': ground_truth,
                'benign_nodes_count': num_events - len(ground_truth),
                'malicious_nodes_count': len(ground_truth)
            })
            sample_id += 1

    # Save synthetic data
    output_file = output_path / 'trec_synthetic_demo.json'
    with open(output_file, 'w') as f:
        json.dump(samples, f, indent=2)

    print(f"Synthetic demo data created: {output_file}")
    print(f"Total samples: {len(samples)}")
    print("Tactic distribution matches paper Table 3")

    return output_file


# Mapping dictionaries (defined outside class for global access)
EntityTypeMapping = {
    'process': 'Process',
    'proc': 'Process',
    'file': 'File',
    'socket': 'Socket',
    'sock': 'Socket',
    'network': 'Socket'
}

EdgeTypeMapping = {
    'write': 'Write',
    'read': 'Read',
    'execute': 'Execute',
    'fork': 'Fork',
    'clone': 'Fork',
    'send': 'Send',
    'recv': 'Receive',
    'receive': 'Receive',
    'mmap': 'Mmap',
    'open': 'Read',
    'close': 'Read'
}


def main():
    parser = argparse.ArgumentParser(
        description='Preprocess TREC APT Tactic Dataset for CAAAPT framework'
    )
    parser.add_argument(
        '--input', '-i',
        type=str,
        default=None,
        help='Path to raw TREC dataset (JSON/JSONL file or directory)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='./data/processed/trec',
        help='Output directory for processed data'
    )
    parser.add_argument(
        '--demo',
        action='store_true',
        help='Create synthetic demo data for testing'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='Maximum number of samples to process (for testing)'
    )
    parser.add_argument(
        '--stats-only',
        action='store_true',
        help='Only show statistics, skip saving'
    )

    args = parser.parse_args()

    if args.demo:
        input_file = create_synthetic_demo(args.output)
        args.input = str(input_file)
        print(f"\nUsing synthetic demo data: {input_file}")

    if not args.input:
        print("Error: Please provide --input path or use --demo flag")
        print("\nUsage examples:")
        print("  python preprocess_trec.py --demo")
        print("  python preprocess_trec.py --input /path/to/trec_data.json --output ./data/processed/trec")
        print("  python preprocess_trec.py --input /path/to/kellect_download --max-samples 100")
        return 1

    # Check if input exists
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input path not found: {input_path}")
        print("\nPlease download the TREC dataset from:")
        print("  https://www.kellect.org/#/kellect-4-aptdataset")
        print("\nOr use --demo flag to create synthetic test data.")
        return 1

    # Initialize processor
    processor = TRECDatasetProcessor(
        data_root=str(input_path.parent) if input_path.is_file() else str(input_path),
        output_dir=args.output
    )

    # Process dataset
    print(f"\n{'=' * 60}")
    print("CAAAPT - TREC Dataset Preprocessor")
    print(f"{'=' * 60}")
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")

    samples = processor.process_dataset(args.input, max_samples=args.max_samples)

    if not samples:
        print("Error: No samples processed. Check input format.")
        return 1

    # Print summary
    processor.print_summary()

    # Save and export
    if not args.stats_only:
        processor.save_processed_data()

        # Create frontend-compatible feature file
        features_path = Path(args.output) / 'frontend_features.csv'
        feature_data = []
        for sample in samples:
            # Extract N-Gram features (simplified for frontend)
            event_types = [e.edge_type for e in sample.events]
            feature_data.append({
                'sample_id': sample.sample_id,
                'tactic': sample.tactic_label,
                'event_count': len(sample.events),
                'unique_event_types': len(set(event_types)),
                'compression_ratio': len(sample.compressed_events) / max(1, len(sample.events)),
                'malicious_ratio': sample.malicious_nodes_count / max(1,
                                                                      sample.benign_nodes_count + sample.malicious_nodes_count)
            })

        pd.DataFrame(feature_data).to_csv(features_path, index=False)
        print(f"Saved frontend features to {features_path}")

    print(f"\n{'=' * 60}")
    print("Preprocessing complete!")
    print(f"{'=' * 60}")

    return 0


if __name__ == "__main__":
    exit(main())