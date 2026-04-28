# src/frontend/compress.py
import hashlib
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple
from collections import defaultdict
from datetime import datetime


@dataclass
class CompressedEvent:
    """Compressed event with frequency count"""
    event_type: str
    source: str
    target: str
    timestamp: int
    frequency: int = 1
    semantics_hash: str = ""

    def __hash__(self):
        return hash((self.event_type, self.source, self.target, self.semantics_hash))


class SemanticCompressor:
    """
    Semantic-preserving data compression as described in Section 3.1.1
    Compresses provenance graph by merging semantically equivalent events
    """

    # Event types that preserve semantics
    SEMANTIC_EVENTS = {'Write', 'Read', 'Mmap', 'Send', 'Receive'}

    def __init__(self):
        self.events = []
        self.event_map = {}
        self.stats = {'original_count': 0, 'compressed_count': 0, 'compression_ratio': 0}

    def compute_semantics_hash(self, event: Dict[str, Any]) -> str:
        """Compute semantic hash for an event based on source semantics"""
        # Extract source semantics (process/file/socket identity)
        source_semantics = f"{event.get('source_type')}:{event.get('source_path', event.get('source_pid'))}"
        target_semantics = f"{event.get('target_type')}:{event.get('target_path', event.get('target_pid'))}"

        # For Write/Read, semantics based on source state
        if event['event_type'] in ['Write', 'Read']:
            semantics_string = f"{event['event_type']}:{source_semantics}"
        else:
            semantics_string = f"{event['event_type']}:{source_semantics}->{target_semantics}"

        return hashlib.md5(semantics_string.encode()).hexdigest()[:16]

    def should_compress(self, prev_event: Dict, curr_event: Dict) -> bool:
        """Determine if two events are semantically equivalent for compression"""
        # Only compress same-type events
        if prev_event['event_type'] != curr_event['event_type']:
            return False

        # Check if source semantics unchanged
        prev_hash = self.compute_semantics_hash(prev_event)
        curr_hash = self.compute_semantics_hash(curr_event)

        return prev_hash == curr_hash

    def compress_sequence(self, events: List[Dict[str, Any]]) -> List[CompressedEvent]:
        """
        Compress event sequence according to Algorithm 1 logic
        Retains only first event when source semantics unchanged, records frequency
        """
        self.stats['original_count'] = len(events)
        compressed = []

        i = 0
        while i < len(events):
            curr = events[i]
            freq = 1
            j = i + 1

            # Merge consecutive semantically equivalent events
            while j < len(events) and self.should_compress(curr, events[j]):
                freq += 1
                j += 1

            # Create compressed event
            compressed_event = CompressedEvent(
                event_type=curr['event_type'],
                source=curr.get('source_path', str(curr.get('source_pid', ''))),
                target=curr.get('target_path', str(curr.get('target_pid', ''))),
                timestamp=curr.get('timestamp', 0),
                frequency=freq,
                semantics_hash=self.compute_semantics_hash(curr)
            )
            compressed.append(compressed_event)
            i = j

        self.stats['compressed_count'] = len(compressed)
        self.stats['compression_ratio'] = len(compressed) / len(events) if events else 1

        return compressed

    def decompress(self, compressed_events: List[CompressedEvent]) -> List[Dict]:
        """Decompress events for reconstruction"""
        decompressed = []
        for ce in compressed_events:
            for _ in range(ce.frequency):
                decompressed.append({
                    'event_type': ce.event_type,
                    'source_path': ce.source,
                    'target_path': ce.target,
                    'timestamp': ce.timestamp
                })
        return decompressed

    def get_stats(self) -> Dict:
        return self.stats


class ProvenanceGraphCompressor:
    """
    Build compressed provenance graph from audit logs
    Following Figure 2 illustration
    """

    def __init__(self):
        self.compressor = SemanticCompressor()
        self.entities = {}  # Track entity state changes
        self.graph = defaultdict(list)

    def process_audit_log(self, log_entries: List[Dict]) -> List[CompressedEvent]:
        """
        Process raw audit logs and build compressed graph
        """
        # Group by entity to track semantic changes
        entity_events = defaultdict(list)

        for entry in log_entries:
            source_id = self._get_entity_id(entry, 'source')
            entity_events[source_id].append(entry)

        all_compressed = []
        for source_id, events in entity_events.items():
            # Sort by timestamp
            events.sort(key=lambda x: x.get('timestamp', 0))
            compressed = self.compressor.compress_sequence(events)
            all_compressed.extend(compressed)

            # Build graph edges
            for ce in compressed:
                self.graph[source_id].append({
                    'target': ce.target,
                    'type': ce.event_type,
                    'freq': ce.frequency,
                    'timestamp': ce.timestamp
                })

        return all_compressed

    def _get_entity_id(self, event: Dict, role: str) -> str:
        """Get unique entity identifier"""
        if role == 'source':
            return event.get('source_path', f"pid:{event.get('source_pid', 'unknown')}")
        else:
            return event.get('target_path', f"pid:{event.get('target_pid', 'unknown')}")

    def get_graph_stats(self) -> Dict:
        return {
            'num_entities': len(self.entities),
            'num_edges': sum(len(edges) for edges in self.graph.values()),
            'compression_ratio': self.compressor.stats['compression_ratio']
        }