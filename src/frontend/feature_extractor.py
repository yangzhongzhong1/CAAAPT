# src/frontend/feature_extractor.py
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from collections import Counter
from typing import List, Dict, Any, Tuple
import re


class NGramFeatureExtractor:
    """
    Feature extraction with N-Gram and TF-IDF 
    """

    def __init__(self, ngram_range=(2, 3), max_features=10000, chi2_pvalue=0.01, top_k=5000):
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.chi2_pvalue = chi2_pvalue
        self.top_k = top_k
        self.tfidf_vectorizer = None
        self.selector = None
        self.selected_features = []

        # ATT&CK pattern keywords for manual supplementation
        self.attack_patterns = {
            'process_injection': ['write_process_memory', 'create_remote_thread', 'queue_user_apc'],
            'credential_dumping': ['lsass', 'sam', 'security', 'registry'],
            'persistence': ['scheduled_task', 'registry_run', 'startup_folder'],
            'defense_evasion': ['delete_log', 'clear_event', 'disable_av'],
            'discovery': ['netstat', 'ipconfig', 'whoami', 'systeminfo', 'tasklist'],
            'lateral_movement': ['psexec', 'wmic', 'schtasks', 'winrm'],
            'privilege_escalation': ['sudo', 'runas', 'bypassuac']
        }

    def extract_ngrams(self, event_sequence: List[Dict]) -> List[str]:
        """Extract Bi-gram and Tri-gram patterns"""
        # Convert events to token strings
        tokens = []
        for event in event_sequence:
            token = f"{event['event_type']}:{event.get('source', '')}->{event.get('target', '')}"
            tokens.append(token)

        ngrams = []

        # Bi-grams (N=2)
        for i in range(len(tokens) - 1):
            ngrams.append(f"BIGRAM:{tokens[i]}|{tokens[i + 1]}")

        # Tri-grams (N=3)
        for i in range(len(tokens) - 2):
            ngrams.append(f"TRIGRAM:{tokens[i]}|{tokens[i + 1]}|{tokens[i + 2]}")

        # Also include individual event types as unigrams
        for token in tokens:
            ngrams.append(f"UNIGRAM:{token}")

        return ngrams

    def _extract_attack_pattern_features(self, event_sequence: List[Dict]) -> Dict:
        """Extract manual features based on ATT&CK patterns"""
        features = {}
        sequence_str = str(event_sequence).lower()

        for tactic, patterns in self.attack_patterns.items():
            count = 0
            for pattern in patterns:
                count += sequence_str.count(pattern)
            features[f"attack_{tactic}"] = min(count, 10) / 10.0  # Normalize

        return features

    def fit_transform(self, sequences: List[List[Dict]], labels: List[int]) -> np.ndarray:
        """Fit extractor and transform sequences to feature vectors"""
        # Extract n-gram strings
        ngram_sequences = []
        for seq in sequences:
            ngrams = self.extract_ngrams(seq)
            ngram_sequences.append(' '.join(ngrams))

        # TF-IDF vectorization (Equation 2)
        self.tfidf_vectorizer = TfidfVectorizer(
            ngram_range=self.ngram_range,
            max_features=self.max_features,
            token_pattern=r'(?u)\b\w+\b'
        )
        tfidf_features = self.tfidf_vectorizer.fit_transform(ngram_sequences)

        # Feature selection with chi-square test (p < 0.01)
        self.selector = SelectKBest(chi2, k=min(self.top_k, tfidf_features.shape[1]))
        selected_features = self.selector.fit_transform(tfidf_features, labels)

        # Get selected feature names
        feature_names = self.tfidf_vectorizer.get_feature_names_out()
        selected_mask = self.selector.get_support()
        self.selected_features = feature_names[selected_mask].tolist()

        # Add manual attack pattern features
        manual_features = []
        for seq in sequences:
            manual_feat = self._extract_attack_pattern_features(seq)
            manual_features.append(list(manual_feat.values()))

        manual_features = np.array(manual_features)

        # Combine TF-IDF features with manual features
        combined = np.hstack([selected_features.toarray(), manual_features])

        return combined

    def transform(self, sequences: List[List[Dict]]) -> np.ndarray:
        """Transform sequences to feature vectors using fitted extractor"""
        ngram_sequences = []
        for seq in sequences:
            ngrams = self.extract_ngrams(seq)
            ngram_sequences.append(' '.join(ngrams))

        tfidf_features = self.tfidf_vectorizer.transform(ngram_sequences)
        selected_features = self.selector.transform(tfidf_features)

        manual_features = []
        for seq in sequences:
            manual_feat = self._extract_attack_pattern_features(seq)
            manual_features.append(list(manual_feat.values()))

        manual_features = np.array(manual_features)

        return np.hstack([selected_features.toarray(), manual_features])