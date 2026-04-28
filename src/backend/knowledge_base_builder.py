"""
CAAAPT Knowledge Base Builder
Builds multi-source security knowledge base: structured knowledge base + vector knowledge base
Sanitized version - Removes sensitive paths, API keys, and internal IPs for submission
"""

import os
import json
import sqlite3
import pickle
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import hashlib

import numpy as np
import pandas as pd
from tqdm import tqdm

# Note: The following imports depend on actual environment configuration
# BERT-related components use placeholder implementation; replace with real models for deployment
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    # Sanitization: provide mock implementation
    SentenceTransformer = None


@dataclass
class StructuredKnowledgeEntry:
    """Structured knowledge entry"""
    entry_id: str
    source_type: str  # 'mitre_attack', 'cti_report', 'ioc', 'historical_case'
    tactic: str
    technique: str
    technique_id: str  # e.g., T1059
    description: str
    indicators: List[str]
    related_cves: List[str]
    created_at: str
    confidence_score: float
    source_reference: str


@dataclass
class VectorKnowledgeEntry:
    """Vector knowledge entry"""
    entry_id: str
    text_content: str
    embedding: List[float]
    metadata: Dict[str, Any]


class StructuredKnowledgeBuilder:
    """
    Structured knowledge base builder
    Stores: ATT&CK tactics/techniques, IOCs, CTI report structured fields
    """

    def __init__(self, db_path: str = "data/knowledge_base/structured.db"):
        self.db_path = db_path
        self._init_database()

    def _init_database(self):
        """Initialize SQLite database schema"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()

        # Create tactic-technique mapping table
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS attack_tactics (
                tactic_id TEXT PRIMARY KEY,
                tactic_name TEXT NOT NULL,
                description TEXT,
                parent_technique_id TEXT,
                technique_name TEXT,
                technique_description TEXT,
                indicators TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create IOC table
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS iocs (
                ioc_id INTEGER PRIMARY KEY AUTOINCREMENT,
                ioc_type TEXT,  -- 'ip', 'domain', 'hash', 'filepath', 'registry'
                ioc_value TEXT,
                tactic_associated TEXT,
                technique_associated TEXT,
                confidence REAL,
                source TEXT,
                first_seen TIMESTAMP,
                last_seen TIMESTAMP,
                UNIQUE(ioc_type, ioc_value)
            )
        """)

        # Create CTI report index table
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS cti_reports (
                report_id TEXT PRIMARY KEY,
                title TEXT,
                publish_date DATE,
                source_org TEXT,
                apt_group TEXT,
                target_sectors TEXT,
                tactics_used TEXT,
                techniques_used TEXT,
                summary TEXT,
                file_path TEXT
            )
        """)

        # Create historical attribution cases table
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS attribution_cases (
                case_id TEXT PRIMARY KEY,
                attack_description TEXT,
                identified_tactics TEXT,
                identified_techniques TEXT,
                confidence REAL,
                verified_by_human BOOLEAN,
                created_at TIMESTAMP
            )
        """)

        # Create hybrid retrieval auxiliary indexes
        self.cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_tactic_tech 
            ON attack_tactics(tactic_name, technique_name)
        """)
        self.cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_ioc_search 
            ON iocs(ioc_value, tactic_associated)
        """)

        self.conn.commit()

    def load_mitre_attack_framework(self, mitre_data_path: str = None):
        """
        Load MITRE ATT&CK framework data
        Sanitized: uses mock data; replace with real MITRE data source for deployment
        """
        # Sanitized: using public ATT&CK tactic list
        tactics = [
            ("TA0001", "Initial Access", "Techniques used to gain entry"),
            ("TA0002", "Execution", "Techniques that run malicious code"),
            ("TA0003", "Persistence", "Techniques to maintain presence"),
            ("TA0004", "Privilege Escalation", "Techniques to gain higher-level permissions"),
            ("TA0005", "Defense Evasion", "Techniques to avoid detection"),
            ("TA0006", "Credential Access", "Techniques to steal credentials"),
            ("TA0007", "Discovery", "Techniques to explore the environment"),
            ("TA0008", "Lateral Movement", "Techniques to move through systems"),
            ("TA0009", "Collection", "Techniques to gather data of interest"),
            ("TA0010", "Exfiltration", "Techniques to steal data"),
            ("TA0011", "Command and Control", "Techniques to communicate with compromised systems")
        ]

        # Sanitized: simplified technique examples
        techniques = {
            "T1059": ("Command and Scripting Interpreter", "Execution"),
            "T1047": ("Windows Management Instrumentation", "Execution"),
            "T1548": ("Abuse Elevation Control Mechanism", "Privilege Escalation"),
            "T1562": ("Impair Defenses", "Defense Evasion"),
            "T1083": ("File and Directory Discovery", "Discovery"),
            "T1021": ("Remote Services", "Lateral Movement")
        }

        for tactic_id, tactic_name, desc in tactics:
            self.cursor.execute("""
                INSERT OR REPLACE INTO attack_tactics 
                (tactic_id, tactic_name, description, created_at)
                VALUES (?, ?, ?, ?)
            """, (tactic_id, tactic_name, desc, datetime.now()))

        for tech_id, (tech_name, tactic_name) in techniques.items():
            self.cursor.execute("""
                UPDATE attack_tactics 
                SET technique_name = ?, technique_description = ?
                WHERE tactic_name = ? AND technique_id IS NULL
            """, (tech_name, f"Technique {tech_id} description", tactic_name))
            # Insert technique record separately
            self.cursor.execute("""
                INSERT OR REPLACE INTO attack_tactics 
                (technique_id, technique_name, tactic_name, created_at)
                VALUES (?, ?, ?, ?)
            """, (tech_id, tech_name, tactic_name, datetime.now()))

        self.conn.commit()
        print(f"[INFO] Loaded MITRE ATT&CK framework: {len(tactics)} tactics, {len(techniques)} techniques")

    def add_cti_report(self, report_data: Dict[str, Any]):
        """Add CTI report"""
        self.cursor.execute("""
            INSERT OR REPLACE INTO cti_reports 
            (report_id, title, publish_date, source_org, apt_group, 
             target_sectors, tactics_used, techniques_used, summary, file_path)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            report_data.get('report_id'),
            report_data.get('title'),
            report_data.get('publish_date'),
            report_data.get('source_org'),
            report_data.get('apt_group'),
            json.dumps(report_data.get('target_sectors', [])),
            json.dumps(report_data.get('tactics_used', [])),
            json.dumps(report_data.get('techniques_used', [])),
            report_data.get('summary'),
            report_data.get('file_path')
        ))
        self.conn.commit()

    def add_ioc(self, ioc_type: str, ioc_value: str, tactic: str,
                technique: str, confidence: float, source: str):
        """Add IOC indicator"""
        try:
            self.cursor.execute("""
                INSERT OR IGNORE INTO iocs 
                (ioc_type, ioc_value, tactic_associated, technique_associated, 
                 confidence, source, first_seen)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (ioc_type, ioc_value, tactic, technique, confidence, source, datetime.now()))
            self.conn.commit()
        except Exception as e:
            print(f"[WARN] Failed to add IOC {ioc_value}: {e}")

    def add_attribution_case(self, case: StructuredKnowledgeEntry):
        """Add historical attribution case"""
        self.cursor.execute("""
            INSERT OR REPLACE INTO attribution_cases 
            (case_id, attack_description, identified_tactics, 
             identified_techniques, confidence, verified_by_human, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            case.entry_id,
            case.description,
            json.dumps([case.tactic, case.technique]),
            json.dumps([]),
            case.confidence_score,
            False,
            datetime.now()
        ))
        self.conn.commit()

    def query_by_tactic(self, tactic: str) -> List[Dict]:
        """Query knowledge by tactic"""
        self.cursor.execute("""
            SELECT * FROM attack_tactics 
            WHERE tactic_name = ? OR technique_name LIKE ?
            LIMIT 50
        """, (tactic, f'%{tactic}%'))
        return [dict(row) for row in self.cursor.fetchall()]

    def query_by_ioc(self, ioc_value: str) -> List[Dict]:
        """Query knowledge by IOC"""
        self.cursor.execute("""
            SELECT * FROM iocs 
            WHERE ioc_value LIKE ?
            ORDER BY confidence DESC
            LIMIT 20
        """, (f'%{ioc_value}%',))
        return [dict(row) for row in self.cursor.fetchall()]

    def close(self):
        self.conn.close()


class VectorKnowledgeBuilder:
    """
    Vector knowledge base builder
    Uses BERT to encode threat intelligence text, supports semantic similarity retrieval
    Sanitized version: provides mock embedder; replace with real BERT model for deployment
    """

    def __init__(self, vector_db_path: str = "data/knowledge_base/vectors.pkl"):
        self.vector_db_path = vector_db_path
        self.entries: List[VectorKnowledgeEntry] = []
        self._init_embedder()
        self._load_existing()

    def _init_embedder(self):
        """Initialize embedding model (sanitized: uses mock or lightweight model)"""
        if SentenceTransformer is not None:
            # Using lightweight model for sanitized demonstration
            # Use domain-adapted BERT for production
            self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
            self.use_real_embedder = True
            print("[INFO] Using real SentenceTransformer for embeddings")
        else:
            # Mock embedder for sanitized testing
            self.use_real_embedder = False
            print("[WARN] SentenceTransformer not available, using mock embedder")

    def _generate_mock_embedding(self, text: str, dim: int = 384) -> List[float]:
        """Generate mock embedding vector (for sanitized testing only)"""
        # Generate deterministic vector using hash
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
                    self.entries = pickle.load(f)
                print(f"[INFO] Loaded {len(self.entries)} existing vector entries")
            except Exception as e:
                print(f"[WARN] Failed to load vector DB: {e}")
                self.entries = []

    def save(self):
        """Save vector database to disk"""
        os.makedirs(os.path.dirname(self.vector_db_path), exist_ok=True)
        with open(self.vector_db_path, 'wb') as f:
            pickle.dump(self.entries, f)
        print(f"[INFO] Saved {len(self.entries)} vector entries to {self.vector_db_path}")

    def add_text(self, text_content: str, metadata: Dict[str, Any]) -> str:
        """Add text entry to vector database"""
        entry_id = hashlib.md5(f"{text_content}_{datetime.now()}".encode()).hexdigest()[:16]
        embedding = self._get_embedding(text_content)

        entry = VectorKnowledgeEntry(
            entry_id=entry_id,
            text_content=text_content,
            embedding=embedding,
            metadata=metadata
        )
        self.entries.append(entry)
        return entry_id

    def add_threat_intelligence(self, ti_data: List[Dict[str, Any]]):
        """Batch add threat intelligence"""
        for item in tqdm(ti_data, desc="Building vector knowledge base"):
            # Build text representation
            text_parts = []
            if 'tactic' in item:
                text_parts.append(f"Tactic: {item['tactic']}")
            if 'technique' in item:
                text_parts.append(f"Technique: {item['technique']}")
            if 'description' in item:
                text_parts.append(f"Description: {item['description']}")
            if 'indicators' in item:
                text_parts.append(f"Indicators: {', '.join(item['indicators'][:5])}")

            text_content = " | ".join(text_parts)

            metadata = {
                'source': item.get('source', 'unknown'),
                'tactic': item.get('tactic', ''),
                'technique': item.get('technique', ''),
                'confidence': item.get('confidence', 0.5),
                'timestamp': datetime.now().isoformat()
            }
            self.add_text(text_content, metadata)

        self.save()

    def similarity_search(self, query: str, top_k: int = 10) -> List[Tuple[VectorKnowledgeEntry, float]]:
        """
        Cosine similarity search for top-k most relevant entries
        """
        query_embedding = self._get_embedding(query)
        query_vec = np.array(query_embedding)

        results = []
        for entry in self.entries:
            entry_vec = np.array(entry.embedding)
            # Cosine similarity calculation
            similarity = np.dot(query_vec, entry_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(entry_vec))
            # Handle numerical errors
            similarity = max(-1.0, min(1.0, similarity))
            results.append((entry, similarity))

        # Sort and return top_k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def search_by_tactic(self, tactic: str, top_k: int = 5) -> List[Dict]:
        """Retrieve by tactic name"""
        results = self.similarity_search(f"attack tactic {tactic}", top_k)
        return [
            {
                'text': r[0].text_content,
                'similarity': r[1],
                'metadata': r[0].metadata
            }
            for r in results
        ]

    def get_statistics(self) -> Dict:
        """Get vector database statistics"""
        return {
            'total_entries': len(self.entries),
            'embedding_dimension': len(self.entries[0].embedding) if self.entries else 0,
            'use_real_embedder': self.use_real_embedder
        }


class KnowledgeBaseManager:
    """
    Knowledge Base Manager
    Integrates structured knowledge base and vector knowledge base, provides unified retrieval interface
    """

    def __init__(self,
                 structured_db_path: str = "data/knowledge_base/structured.db",
                 vector_db_path: str = "data/knowledge_base/vectors.pkl"):
        self.structured = StructuredKnowledgeBuilder(structured_db_path)
        self.vector = VectorKnowledgeBuilder(vector_db_path)
        self.retrieval_cache = {}  # Simple cache
        print("[INFO] Knowledge Base Manager initialized")

    def hybrid_retrieval(self,
                         query: str,
                         tactic_filter: Optional[str] = None,
                         top_k: int = 10,
                         alpha: float = 0.6) -> List[Dict]:
        """
        Hybrid retrieval: combines structured query and semantic vector retrieval
        alpha: weight for structured retrieval, vector retrieval weight = 1-alpha
        """
        results = []

        # 1. Structured retrieval
        structured_results = []
        if tactic_filter:
            structured_results = self.structured.query_by_tactic(tactic_filter)
            for r in structured_results:
                results.append({
                    'source': 'structured',
                    'content': r.get('description', '') or r.get('technique_description', ''),
                    'tactic': r.get('tactic_name', tactic_filter),
                    'technique': r.get('technique_name', ''),
                    'score': alpha,
                    'metadata': r
                })

        # 2. Semantic vector retrieval
        vector_results = self.vector.similarity_search(query, top_k=top_k)
        for entry, similarity in vector_results:
            results.append({
                'source': 'vector',
                'content': entry.text_content,
                'tactic': entry.metadata.get('tactic', ''),
                'technique': entry.metadata.get('technique', ''),
                'score': (1 - alpha) * similarity,
                'similarity_raw': similarity,
                'metadata': entry.metadata
            })

        # Sort by composite score
        results.sort(key=lambda x: x['score'], reverse=True)

        return results[:top_k]

    def get_knowledge_for_attack_pattern(self,
                                         event_description: str,
                                         tactics: List[str]) -> Dict:
        """
        Retrieve relevant knowledge for specific attack event
        Used for constructing RAG context
        """
        context = {
            'structured_knowledge': [],
            'semantic_knowledge': [],
            'related_iocs': []
        }

        # Retrieve relevant knowledge for each tactic
        for tactic in tactics:
            structured = self.structured.query_by_tactic(tactic)
            context['structured_knowledge'].extend(structured)

            semantic = self.vector.search_by_tactic(tactic, top_k=3)
            context['semantic_knowledge'].extend(semantic)

        # Deduplication
        context['structured_knowledge'] = list({
                                                   d.get('technique_id'): d for d in context['structured_knowledge']
                                                   if d.get('technique_id')
                                               }.values())

        context['semantic_knowledge'] = list({
                                                 d['text']: d for d in context['semantic_knowledge']
                                             }.values())

        return context

    def build_from_cti_reports(self, cti_data_path: str):
        """
        Build knowledge base from CTI reports
        cti_data_path: directory path containing JSON format CTI reports
        """
        if not os.path.exists(cti_data_path):
            print(f"[WARN] CTI data path not found: {cti_data_path}")
            print("[INFO] Using built-in MITRE ATT&CK data instead")
            self.structured.load_mitre_attack_framework()
            return

        # Load CTI reports (sanitized: display file list instead of actual content)
        report_files = [f for f in os.listdir(cti_data_path) if f.endswith('.json')]
        print(f"[INFO] Found {len(report_files)} CTI report files")

        ti_data_for_vector = []

        for report_file in tqdm(report_files[:100], desc="Processing CTI reports"):
            file_path = os.path.join(cti_data_path, report_file)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    report = json.load(f)

                # Add to structured database
                self.structured.add_cti_report({
                    'report_id': report.get('id', report_file),
                    'title': report.get('title', ''),
                    'publish_date': report.get('date', ''),
                    'source_org': report.get('source', ''),
                    'apt_group': report.get('apt_group', ''),
                    'target_sectors': report.get('target_sectors', []),
                    'tactics_used': report.get('tactics', []),
                    'techniques_used': report.get('techniques', []),
                    'summary': report.get('summary', ''),
                    'file_path': file_path
                })

                # Prepare vector database data
                for tactic in report.get('tactics', []):
                    ti_data_for_vector.append({
                        'source': report.get('source', 'CTI'),
                        'tactic': tactic,
                        'description': report.get('summary', ''),
                        'indicators': report.get('indicators', []),
                        'confidence': report.get('confidence', 0.7)
                    })

            except Exception as e:
                print(f"[WARN] Failed to process {report_file}: {e}")

        # Build vector database
        if ti_data_for_vector:
            self.vector.add_threat_intelligence(ti_data_for_vector)

        # Supplement with MITRE data
        self.structured.load_mitre_attack_framework()

        print(f"[INFO] Knowledge base built: {len(ti_data_for_vector)} vector entries")

    def get_statistics(self) -> Dict:
        """Get knowledge base statistics"""
        # Get structured database statistics
        structured_stats = {}
        try:
            cursor = self.structured.conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM attack_tactics")
            structured_stats['tactics_count'] = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM iocs")
            structured_stats['iocs_count'] = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM cti_reports")
            structured_stats['reports_count'] = cursor.fetchone()[0]
        except Exception as e:
            structured_stats = {'error': str(e)}

        return {
            'structured': structured_stats,
            'vector': self.vector.get_statistics(),
            'cache_size': len(self.retrieval_cache)
        }

    def close(self):
        """Close knowledge base connection"""
        self.structured.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def build_knowledge_base(
        cti_reports_path: Optional[str] = None,
        output_dir: str = "data/knowledge_base"
):
    """
    Main build function
    Usage example:
        build_knowledge_base(cti_reports_path="./data/raw/cti_reports")
    """
    print("=" * 60)
    print("CAAAPT Knowledge Base Builder")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)

    with KnowledgeBaseManager(
            structured_db_path=os.path.join(output_dir, "structured.db"),
            vector_db_path=os.path.join(output_dir, "vectors.pkl")
    ) as kb:

        # Build knowledge base
        if cti_reports_path and os.path.exists(cti_reports_path):
            kb.build_from_cti_reports(cti_reports_path)
        else:
            print(f"[INFO] CTI reports not found at {cti_reports_path}, using MITRE data only")
            kb.structured.load_mitre_attack_framework()

        # Add sample IOCs (sanitized)
        sample_iocs = [
            ("ip", "192.168.x.x", "Command and Control", "T1071", 0.85, "public_threat_intel"),
            ("hash", "44d88612fea8a8f36de82e1278abb02f", "Execution", "T1059", 0.92, "public_threat_intel"),
            ("filepath", "/tmp/malicious.sh", "Persistence", "T1543", 0.78, "public_threat_intel"),
        ]
        for ioc_type, ioc_value, tactic, technique, conf, source in sample_iocs:
            kb.structured.add_ioc(ioc_type, ioc_value, tactic, technique, conf, source)

        stats = kb.get_statistics()
        print("\n[INFO] Knowledge Base Statistics:")
        print(json.dumps(stats, indent=2))

        # Test retrieval
        test_results = kb.hybrid_retrieval(
            query="attacker using PowerShell to download malicious payload",
            tactic_filter="Execution",
            top_k=5
        )
        print(f"\n[INFO] Test retrieval returned {len(test_results)} results")

    print(f"\n[INFO] Knowledge base built successfully at {output_dir}")


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="CAAAPT Knowledge Base Builder")
    parser.add_argument("--cti_path", type=str, default=None,
                        help="Path to CTI reports directory (JSON format)")
    parser.add_argument("--output_dir", type=str, default="data/knowledge_base",
                        help="Output directory for knowledge base")

    args = parser.parse_args()

    build_knowledge_base(
        cti_reports_path=args.cti_path,
        output_dir=args.output_dir
    )