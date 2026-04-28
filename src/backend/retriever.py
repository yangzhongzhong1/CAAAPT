"""
CAAAPT Hybrid Retriever
Multi-modal fusion retrieval mechanism combining structured SQL queries and semantic vector retrieval
Sanitized version for submission - Removes sensitive paths, API keys, and internal IPs
"""

import os
import json
import sqlite3
import pickle
from typing import List, Dict, Any, Tuple, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import hashlib
import re

import numpy as np
from tqdm import tqdm

# Note: Import for vector similarity; sentence-transformers is optional
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None


@dataclass
class RetrievalResult:
    """Unified retrieval result structure"""
    content: str
    source: str  # 'structured_sql', 'vector_semantic', 'hybrid'
    score: float
    tactic: Optional[str] = None
    technique: Optional[str] = None
    technique_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    similarity_raw: Optional[float] = None


class StructuredSQLRetriever:
    """
    Structured log and knowledge retriever using SQL
    Handles audit logs, process trees, network records with multi-dimensional indexes
    """

    def __init__(self, db_path: str = "data/knowledge_base/structured.db"):
        self.db_path = db_path
        self.conn = None
        self._connect()
        self._ensure_indexes()

    def _connect(self):
        """Establish database connection"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self.cursor = self.conn.cursor()

    def _ensure_indexes(self):
        """Ensure multi-dimensional indexes for efficient queries"""
        # Time-series optimized indexes
        self.cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_event_timestamp 
            ON audit_events(timestamp)
        """)
        self.cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_entity_identifier 
            ON audit_events(entity_id)
        """)
        self.cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_process_tree_parent 
            ON process_tree(parent_pid)
        """)
        self.cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_network_connection 
            ON network_connections(src_ip, dst_ip)
        """)
        self.conn.commit()

    def init_audit_schema(self):
        """Initialize audit log table schema for provenance graph storage"""
        # Audit events table
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS audit_events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp INTEGER,
                event_type TEXT,
                entity_type TEXT,
                entity_id TEXT,
                source_entity TEXT,
                target_entity TEXT,
                attributes TEXT,
                compressed_frequency INTEGER DEFAULT 1
            )
        """)

        # Process tree table
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS process_tree (
                process_id TEXT PRIMARY KEY,
                process_name TEXT,
                parent_pid TEXT,
                cmd_line TEXT,
                user_id TEXT,
                start_time INTEGER,
                end_time INTEGER
            )
        """)

        # Network connections table
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS network_connections (
                conn_id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp INTEGER,
                src_ip TEXT,
                src_port INTEGER,
                dst_ip TEXT,
                dst_port INTEGER,
                protocol TEXT,
                process_id TEXT,
                bytes_sent INTEGER,
                bytes_recv INTEGER
            )
        """)

        self.conn.commit()
        print("[INFO] Audit schema initialized")

    def insert_audit_event(self, event: Dict[str, Any]):
        """Insert a single audit event"""
        self.cursor.execute("""
            INSERT INTO audit_events 
            (timestamp, event_type, entity_type, entity_id, source_entity, 
             target_entity, attributes, compressed_frequency)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            event.get('timestamp'),
            event.get('event_type'),
            event.get('entity_type'),
            event.get('entity_id'),
            event.get('source_entity'),
            event.get('target_entity'),
            json.dumps(event.get('attributes', {})),
            event.get('compressed_frequency', 1)
        ))
        self.conn.commit()

    def batch_insert_audit_events(self, events: List[Dict[str, Any]]):
        """Batch insert audit events for efficiency"""
        batch_data = [
            (
                e.get('timestamp'),
                e.get('event_type'),
                e.get('entity_type'),
                e.get('entity_id'),
                e.get('source_entity'),
                e.get('target_entity'),
                json.dumps(e.get('attributes', {})),
                e.get('compressed_frequency', 1)
            )
            for e in events
        ]
        self.cursor.executemany("""
            INSERT INTO audit_events 
            (timestamp, event_type, entity_type, entity_id, source_entity, 
             target_entity, attributes, compressed_frequency)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, batch_data)
        self.conn.commit()

    def query_by_time_window(self, start_time: int, end_time: int,
                             limit: int = 1000) -> List[Dict]:
        """
        Exact time-window query with SQL
        Enables efficient retrieval of events within specified time range
        """
        self.cursor.execute("""
            SELECT * FROM audit_events 
            WHERE timestamp BETWEEN ? AND ?
            ORDER BY timestamp
            LIMIT ?
        """, (start_time, end_time, limit))
        return [dict(row) for row in self.cursor.fetchall()]

    def query_by_entity_id(self, entity_id: str, limit: int = 500) -> List[Dict]:
        """Query all events associated with a specific entity"""
        self.cursor.execute("""
            SELECT * FROM audit_events 
            WHERE entity_id = ? OR source_entity = ? OR target_entity = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """, (entity_id, entity_id, entity_id, limit))
        return [dict(row) for row in self.cursor.fetchall()]

    def trace_process_tree(self, process_id: str, max_depth: int = 10) -> List[Dict]:
        """
        Trace process ancestry using recursive CTE
        Enables provenance analysis and attack chain reconstruction
        """
        try:
            # Recursive CTE for process tree traversal
            query = f"""
                WITH RECURSIVE process_ancestry AS (
                    SELECT * FROM process_tree WHERE process_id = ?
                    UNION ALL
                    SELECT p.* FROM process_tree p
                    INNER JOIN process_ancestry pa ON p.process_id = pa.parent_pid
                    WHERE pa.parent_pid IS NOT NULL AND pa.parent_pid != ''
                    LIMIT {max_depth}
                )
                SELECT * FROM process_ancestry
            """
            self.cursor.execute(query, (process_id,))
            return [dict(row) for row in self.cursor.fetchall()]
        except Exception as e:
            print(f"[WARN] Process tree trace failed: {e}")
            return []

    def join_file_operations_by_process(self, process_id: str) -> List[Dict]:
        """
        Multi-table join: trace file operations by process ID
        Example of complex relational query for provenance analysis
        """
        self.cursor.execute("""
            SELECT a.timestamp, a.event_type, a.source_entity, a.target_entity,
                   p.process_name, p.cmd_line
            FROM audit_events a
            LEFT JOIN process_tree p ON a.entity_id = p.process_id
            WHERE a.entity_id = ? AND a.event_type IN ('Write', 'Read', 'Mmap')
            ORDER BY a.timestamp
        """, (process_id,))
        return [dict(row) for row in self.cursor.fetchall()]

    def query_knowledge_by_tactic(self, tactic: str) -> List[Dict]:
        """Query structured knowledge base by tactic"""
        self.cursor.execute("""
            SELECT * FROM attack_tactics 
            WHERE tactic_name = ? OR technique_name LIKE ?
        """, (tactic, f'%{tactic}%'))
        return [dict(row) for row in self.cursor.fetchall()]

    def query_iocs_by_type(self, ioc_type: str) -> List[Dict]:
        """Query IOCs by type (ip, domain, hash, filepath)"""
        self.cursor.execute("""
            SELECT * FROM iocs 
            WHERE ioc_type = ?
            ORDER BY confidence DESC
        """, (ioc_type,))
        return [dict(row) for row in self.cursor.fetchall()]

    def exact_match_ioc(self, ioc_value: str) -> Optional[Dict]:
        """Exact match IOC lookup"""
        self.cursor.execute("""
            SELECT * FROM iocs 
            WHERE ioc_value = ?
        """, (ioc_value,))
        row = self.cursor.fetchone()
        return dict(row) if row else None

    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()


class VectorSemanticRetriever:
    """
    Semantic vector retriever using BERT embeddings
    Supports cosine similarity search on unstructured text
    """

    def __init__(self, vector_db_path: str = "data/knowledge_base/vectors.pkl"):
        self.vector_db_path = vector_db_path
        self.entries: List[Tuple[str, str, List[float], Dict]] = []  # (id, text, embedding, metadata)
        self._init_embedder()
        self._load_existing()

    def _init_embedder(self):
        """Initialize BERT-based embedder for cybersecurity domain"""
        if SentenceTransformer is not None:
            # Using MiniLM for efficiency; domain-adapted BERT recommended for production
            self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
            self.use_real_embedder = True
            print("[INFO] Vector retriever using real SentenceTransformer")
        else:
            self.use_real_embedder = False
            print("[WARN] SentenceTransformer not available, using mock embeddings")

    def _generate_mock_embedding(self, text: str, dim: int = 384) -> List[float]:
        """Generate deterministic mock embedding (for testing only)"""
        hash_val = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
        np.random.seed(hash_val)
        embedding = np.random.randn(dim)
        embedding = embedding / np.linalg.norm(embedding)
        return embedding.tolist()

    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding vector for text"""
        if self.use_real_embedder:
            return self.embedder.encode(text).tolist()
        else:
            return self._generate_mock_embedding(text)

    def _load_existing(self):
        """Load existing vector database"""
        if os.path.exists(self.vector_db_path):
            try:
                with open(self.vector_db_path, 'rb') as f:
                    data = pickle.load(f)
                    # Handle different data formats
                    if isinstance(data, list):
                        self.entries = data
                    elif isinstance(data, dict) and 'entries' in data:
                        self.entries = data['entries']
                    else:
                        self.entries = []
                print(f"[INFO] Loaded {len(self.entries)} existing vector entries")
            except Exception as e:
                print(f"[WARN] Failed to load vector DB: {e}")
                self.entries = []

    def save(self):
        """Save vector database to disk"""
        os.makedirs(os.path.dirname(self.vector_db_path), exist_ok=True)
        with open(self.vector_db_path, 'wb') as f:
            pickle.dump(self.entries, f)
        print(f"[INFO] Saved {len(self.entries)} vector entries")

    def add_entry(self, text: str, metadata: Dict[str, Any]) -> str:
        """Add a text entry to the vector database"""
        entry_id = hashlib.md5(f"{text}_{datetime.now()}".encode()).hexdigest()[:16]
        embedding = self._get_embedding(text)
        self.entries.append((entry_id, text, embedding, metadata))
        return entry_id

    def add_threat_intelligence_entries(self, ti_texts: List[Tuple[str, Dict]]):
        """Batch add threat intelligence entries"""
        for text, metadata in tqdm(ti_texts, desc="Adding vector entries"):
            self.add_entry(text, metadata)
        self.save()

    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Compute cosine similarity between two vectors"""
        v1 = np.array(vec1)
        v2 = np.array(vec2)
        dot = np.dot(v1, v2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        similarity = dot / (norm1 * norm2)
        return max(-1.0, min(1.0, similarity))

    def similarity_search(self, query: str, top_k: int = 10) -> List[Tuple[str, str, float, Dict]]:
        """
        Perform cosine similarity search for top-k relevant entries
        Implements approximate nearest neighbor via brute force (for demonstration)
        For production: use HNSW or FAISS indexing
        """
        if not self.entries:
            return []

        query_embedding = self._get_embedding(query)

        results = []
        for entry_id, text, embedding, metadata in self.entries:
            similarity = self.cosine_similarity(query_embedding, embedding)
            results.append((entry_id, text, similarity, metadata))

        # Sort by similarity descending
        results.sort(key=lambda x: x[2], reverse=True)
        return results[:top_k]

    def search_by_tactic(self, tactic: str, top_k: int = 5) -> List[Tuple[str, str, float, Dict]]:
        """Search for entries related to a specific tactic"""
        query = f"ATT&CK tactic {tactic} cyber attack technique"
        return self.similarity_search(query, top_k)

    def search_by_technique(self, technique: str, top_k: int = 5) -> List[Tuple[str, str, float, Dict]]:
        """Search for entries related to a specific technique"""
        query = f"ATT&CK technique {technique} adversary behavior"
        return self.similarity_search(query, top_k)

    def search_by_attack_pattern(self, event_description: str, top_k: int = 10) -> List[Tuple[str, str, float, Dict]]:
        """Search for similar attack patterns based on event description"""
        return self.similarity_search(event_description, top_k)

    def get_statistics(self) -> Dict:
        """Get vector database statistics"""
        return {
            'total_entries': len(self.entries),
            'embedding_dimension': len(self.entries[0][2]) if self.entries else 0,
            'use_real_embedder': self.use_real_embedder
        }


class HybridRetriever:
    """
    Hybrid retrieval mechanism combining structured SQL queries and semantic vector retrieval
    Implements linear weighted fusion as described in Section 3.2.2 of the paper
    """

    def __init__(self,
                 structured_db_path: str = "data/knowledge_base/structured.db",
                 vector_db_path: str = "data/knowledge_base/vectors.pkl"):
        self.structured_retriever = StructuredSQLRetriever(structured_db_path)
        self.vector_retriever = VectorSemanticRetriever(vector_db_path)
        self.retrieval_cache: Dict[str, List[RetrievalResult]] = {}
        self.cache_ttl_seconds = 300  # 5 minutes cache TTL

    def _get_cache_key(self, query: str, tactic_filter: Optional[str], top_k: int, alpha: float) -> str:
        """Generate cache key for retrieval results"""
        key_str = f"{query}_{tactic_filter}_{top_k}_{alpha}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def _normalize_scores(self, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """Normalize scores to [0, 1] range for fair fusion"""
        if not results:
            return results

        scores = [r.score for r in results]
        min_score, max_score = min(scores), max(scores)

        if max_score > min_score:
            for r in results:
                r.score = (r.score - min_score) / (max_score - min_score)
        else:
            for r in results:
                r.score = 0.5

        return results

    def structured_retrieval(self,
                             query: str,
                             tactic_filter: Optional[str] = None,
                             top_k: int = 10) -> List[RetrievalResult]:
        """
        Perform structured SQL-based retrieval
        """
        results = []

        # Query by tactic if filter provided
        if tactic_filter:
            tactic_results = self.structured_retriever.query_knowledge_by_tactic(tactic_filter)
            for r in tactic_results[:top_k]:
                results.append(RetrievalResult(
                    content=r.get('description', '') or r.get('technique_description', ''),
                    source='structured_sql',
                    score=0.5,  # Base score, will be normalized later
                    tactic=r.get('tactic_name', tactic_filter),
                    technique=r.get('technique_name'),
                    technique_id=r.get('technique_id'),
                    metadata=r
                ))

        # IOC exact match (highest precision)
        # Try to extract potential IOC patterns from query
        ip_pattern = r'\b(?:\d{1,3}\.){3}\d{1,3}\b'
        hash_pattern = r'\b[a-fA-F0-9]{32}\b'

        ip_matches = re.findall(ip_pattern, query)
        for ip in ip_matches:
            ioc_result = self.structured_retriever.exact_match_ioc(ip)
            if ioc_result:
                results.append(RetrievalResult(
                    content=f"IOC: {ioc_result.get('ioc_type')} = {ioc_result.get('ioc_value')}",
                    source='structured_sql',
                    score=1.0,  # Exact match gets high score
                    tactic=ioc_result.get('tactic_associated'),
                    technique=ioc_result.get('technique_associated'),
                    metadata=ioc_result
                ))

        return results

    def vector_retrieval(self,
                         query: str,
                         top_k: int = 10) -> List[RetrievalResult]:
        """
        Perform semantic vector-based retrieval
        """
        results = []
        vector_results = self.vector_retriever.similarity_search(query, top_k=top_k)

        for entry_id, text, similarity, metadata in vector_results:
            results.append(RetrievalResult(
                content=text,
                source='vector_semantic',
                score=similarity,
                tactic=metadata.get('tactic'),
                technique=metadata.get('technique'),
                similarity_raw=similarity,
                metadata=metadata
            ))

        return results

    def hybrid_retrieval(self,
                         query: str,
                         tactic_filter: Optional[str] = None,
                         top_k: int = 10,
                         alpha: float = 0.6,
                         use_cache: bool = True) -> List[RetrievalResult]:
        """
        Hybrid retrieval with linear weighted fusion

        Args:
            query: Search query string
            tactic_filter: Optional tactic name for filtering
            top_k: Number of top results to return
            alpha: Weight for structured retrieval (1-alpha for vector retrieval)
            use_cache: Whether to use cached results

        Returns:
            List of RetrievalResult objects sorted by fused score
        """
        # Check cache
        cache_key = self._get_cache_key(query, tactic_filter, top_k, alpha)
        if use_cache and cache_key in self.retrieval_cache:
            cached_results = self.retrieval_cache[cache_key]
            if len(cached_results) >= top_k:
                return cached_results[:top_k]

        # Perform retrievals
        structured_results = self.structured_retrieval(query, tactic_filter, top_k * 2)
        vector_results = self.vector_retrieval(query, top_k * 2)

        # Apply weights and combine
        for r in structured_results:
            r.score = r.score * alpha
            r.source = f"hybrid(weight={alpha})"

        for r in vector_results:
            r.score = r.score * (1 - alpha)
            r.source = f"hybrid(weight={1 - alpha})"

        # Combine and sort
        all_results = structured_results + vector_results
        all_results.sort(key=lambda x: x.score, reverse=True)

        # Deduplicate by content (keep highest score)
        seen_contents = set()
        deduplicated = []
        for r in all_results:
            content_hash = hashlib.md5(r.content.encode()).hexdigest()[:16]
            if content_hash not in seen_contents:
                seen_contents.add(content_hash)
                deduplicated.append(r)

        # Cache results
        if use_cache:
            self.retrieval_cache[cache_key] = deduplicated
            # Simple cache cleanup (keep only recent)
            if len(self.retrieval_cache) > 100:
                # Remove oldest 20 entries
                keys_to_remove = list(self.retrieval_cache.keys())[:20]
                for k in keys_to_remove:
                    del self.retrieval_cache[k]

        return deduplicated[:top_k]

    def retrieve_provenance_context(self,
                                    entity_id: str,
                                    time_window_minutes: int = 30) -> Dict:
        """
        Retrieve provenance graph context for a specific entity
        Combines structured process tree queries with semantic IOC lookup
        """
        context = {
            'events': [],
            'process_tree': [],
            'network_connections': [],
            'ioc_matches': []
        }

        # Get events for this entity
        events = self.structured_retriever.query_by_entity_id(entity_id, limit=200)
        context['events'] = events

        # Trace process ancestry
        if events:
            # Extract process ID from events
            process_ids = set()
            for e in events:
                if e.get('entity_id'):
                    process_ids.add(e['entity_id'])
                if e.get('source_entity'):
                    process_ids.add(e['source_entity'])

            for pid in list(process_ids)[:5]:  # Limit for performance
                tree = self.structured_retriever.trace_process_tree(pid)
                context['process_tree'].extend(tree)

        return context

    def retrieve_rag_context(self,
                             suspicious_log: str,
                             tactics: List[str],
                             top_k: int = 5) -> Dict:
        """
        Retrieve comprehensive context for RAG-based LLM attribution
        This is the main retrieval method used by the backend LLM (Section 3.2.2)
        """
        context = {
            'structured_knowledge': [],
            'semantic_knowledge': [],
            'related_iocs': [],
            'attack_patterns': []
        }

        # Retrieve for each tactic
        for tactic in tactics:
            tactic_results = self.hybrid_retrieval(
                query=suspicious_log,
                tactic_filter=tactic,
                top_k=top_k,
                alpha=0.6
            )

            for r in tactic_results:
                entry = {
                    'content': r.content,
                    'tactic': r.tactic,
                    'technique': r.technique,
                    'score': r.score,
                    'source': r.source
                }

                if r.source.startswith('hybrid') and r.tactic:
                    context['structured_knowledge'].append(entry)
                else:
                    context['semantic_knowledge'].append(entry)

        # Deduplicate
        context['structured_knowledge'] = list({
                                                   f"{e['tactic']}_{e['technique']}": e
                                                   for e in context['structured_knowledge']
                                               }.values())

        context['semantic_knowledge'] = list({
                                                 e['content'][:100]: e
                                                 for e in context['semantic_knowledge']
                                             }.values())

        return context

    def batch_retrieve(self,
                       queries: List[str],
                       top_k: int = 5) -> List[List[RetrievalResult]]:
        """Batch retrieval for multiple queries"""
        results = []
        for query in queries:
            query_results = self.hybrid_retrieval(query, top_k=top_k)
            results.append(query_results)
        return results

    def get_statistics(self) -> Dict:
        """Get retriever statistics"""
        return {
            'structured': {
                'db_path': self.structured_retriever.db_path
            },
            'vector': self.vector_retriever.get_statistics(),
            'cache_size': len(self.retrieval_cache)
        }

    def close(self):
        """Close all resources"""
        self.structured_retriever.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# Convenience function for quick retrieval
def create_retriever(structured_db_path: str = "data/knowledge_base/structured.db",
                     vector_db_path: str = "data/knowledge_base/vectors.pkl") -> HybridRetriever:
    """Factory function to create and initialize the hybrid retriever"""
    return HybridRetriever(structured_db_path, vector_db_path)


# Example usage and test function
def test_retriever():
    """Test function for verification"""
    print("=" * 60)
    print("Testing CAAAPT Hybrid Retriever")
    print("=" * 60)

    with create_retriever() as retriever:
        # Initialize schema
        retriever.structured_retriever.init_audit_schema()

        # Test hybrid retrieval
        test_query = "PowerShell downloading malicious payload from suspicious domain"

        print(f"\n[TEST] Query: {test_query}")
        results = retriever.hybrid_retrieval(
            query=test_query,
            tactic_filter="Execution",
            top_k=5,
            alpha=0.6
        )

        print(f"\n[RESULTS] Retrieved {len(results)} items:")
        for i, r in enumerate(results):
            print(f"\n  [{i + 1}] Source: {r.source}")
            print(f"      Score: {r.score:.4f}")
            print(f"      Tactic: {r.tactic}")
            print(f"      Content: {r.content[:150]}...")

        # Test RAG context retrieval
        print("\n[TEST] RAG Context Retrieval")
        rag_context = retriever.retrieve_rag_context(
            suspicious_log="Process created with cmd.exe /c powershell -enc base64...",
            tactics=["Execution", "Defense Evasion"],
            top_k=3
        )

        print(f"  Structured knowledge: {len(rag_context['structured_knowledge'])} items")
        print(f"  Semantic knowledge: {len(rag_context['semantic_knowledge'])} items")

        stats = retriever.get_statistics()
        print(f"\n[STATISTICS] {json.dumps(stats, indent=2)}")

    print("\n[INFO] Retriever test completed")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CAAAPT Hybrid Retriever")
    parser.add_argument("--test", action="store_true", help="Run test")
    parser.add_argument("--structured_db", type=str, default="data/knowledge_base/structured.db",
                        help="Path to structured database")
    parser.add_argument("--vector_db", type=str, default="data/knowledge_base/vectors.pkl",
                        help="Path to vector database")

    args = parser.parse_args()

    if args.test:
        test_retriever()
    else:
        # Interactive mode
        retriever = create_retriever(args.structured_db, args.vector_db)
        print("CAAAPT Hybrid Retriever initialized. Type 'quit' to exit.")

        while True:
            query = input("\nEnter search query: ")
            if query.lower() in ['quit', 'exit', 'q']:
                break

            results = retriever.hybrid_retrieval(query, top_k=5)
            print(f"\nFound {len(results)} results:")
            for i, r in enumerate(results):
                print(f"\n[{i + 1}] Score: {r.score:.4f} | Tactic: {r.tactic}")
                print(f"    {r.content[:200]}...")

        retriever.close()