"""
Lightweight fastText Classifier for News Headline Classification.
Uses quantized INT8 weights and drops raw text immediately after feature extraction.
Designed for minimal RAM usage (<14GB constraint) with no heavy NLP models.
"""

import os
import re
import struct
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import math


@dataclass
class ClassificationResult:
    """Classification result container."""
    label: str           # BULLISH, BEARISH, or NEUTRAL
    confidence: float    # Confidence score (0-1)
    scores: Dict[str, float]  # Raw scores for all labels
    
    @property
    def is_decisive(self) -> bool:
        """Check if classification is confident enough for trading."""
        return self.confidence > 0.6


class QuantizedFastTextModel:
    """
    Minimal fastText-like model with INT8 quantization.
    Implements bag-of-words with linear classifier.
    Memory optimized for <10MB footprint.
    """
    
    LABELS = ["BULLISH", "BEARISH", "NEUTRAL"]
    
    def __init__(self):
        self._vocab: Dict[str, int] = {}
        self._weights: Dict[int, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self._biases: Dict[str, float] = {label: 0.0 for label in self.LABELS}
        self._trained = False
        
        # Build minimal embedded model
        self._build_embedded_model()
    
    def _build_embedded_model(self):
        """Build embedded quantized model weights."""
        # Feature words with their sentiment weights (simulating trained fastText)
        features = {
            # Bullish indicators
            "surge": {"BULLISH": 0.8, "BEARISH": -0.3, "NEUTRAL": -0.2},
            "rally": {"BULLISH": 0.75, "BEARISH": -0.3, "NEUTRAL": -0.2},
            "soar": {"BULLISH": 0.85, "BEARISH": -0.3, "NEUTRAL": -0.25},
            "jump": {"BULLISH": 0.6, "BEARISH": -0.2, "NEUTRAL": -0.1},
            "climb": {"BULLISH": 0.55, "BEARISH": -0.2, "NEUTRAL": -0.1},
            "gain": {"BULLISH": 0.7, "BEARISH": -0.25, "NEUTRAL": -0.15},
            "profits": {"BULLISH": 0.75, "BEARISH": -0.3, "NEUTRAL": -0.15},
            "beat": {"BULLISH": 0.65, "BEARISH": -0.2, "NEUTRAL": -0.15},
            "outperform": {"BULLISH": 0.7, "BEARISH": -0.25, "NEUTRAL": -0.15},
            "bullish": {"BULLISH": 0.9, "BEARISH": -0.4, "NEUTRAL": -0.25},
            "optimistic": {"BULLISH": 0.65, "BEARISH": -0.25, "NEUTRAL": -0.15},
            "confidence": {"BULLISH": 0.55, "BEARISH": -0.2, "NEUTRAL": -0.1},
            "strong": {"BULLISH": 0.5, "BEARISH": -0.2, "NEUTRAL": -0.1},
            "record": {"BULLISH": 0.6, "BEARISH": -0.15, "NEUTRAL": -0.15},
            "high": {"BULLISH": 0.4, "BEARISH": -0.15, "NEUTRAL": 0.0},
            "higher": {"BULLISH": 0.5, "BEARISH": -0.2, "NEUTRAL": -0.1},
            "breakthrough": {"BULLISH": 0.7, "BEARISH": -0.25, "NEUTRAL": -0.15},
            "breakout": {"BULLISH": 0.75, "BEARISH": -0.25, "NEUTRAL": -0.2},
            "moon": {"BULLISH": 0.85, "BEARISH": -0.3, "NEUTRAL": -0.25},
            "rocket": {"BULLISH": 0.8, "BEARISH": -0.3, "NEUTRAL": -0.2},
            
            # Bearish indicators
            "crash": {"BULLISH": -0.4, "BEARISH": 0.9, "NEUTRAL": -0.25},
            "plunge": {"BULLISH": -0.35, "BEARISH": 0.85, "NEUTRAL": -0.2},
            "dump": {"BULLISH": -0.35, "BEARISH": 0.8, "NEUTRAL": -0.2},
            "tumble": {"BULLISH": -0.3, "BEARISH": 0.75, "NEUTRAL": -0.2},
            "fall": {"BULLISH": -0.3, "BEARISH": 0.65, "NEUTRAL": -0.15},
            "drop": {"BULLISH": -0.3, "BEARISH": 0.6, "NEUTRAL": -0.15},
            "decline": {"BULLISH": -0.35, "BEARISH": 0.7, "NEUTRAL": -0.15},
            "loss": {"BULLISH": -0.3, "BEARISH": 0.75, "NEUTRAL": -0.2},
            "losses": {"BULLISH": -0.35, "BEARISH": 0.8, "NEUTRAL": -0.2},
            "bearish": {"BULLISH": -0.4, "BEARISH": 0.9, "NEUTRAL": -0.25},
            "pessimistic": {"BULLISH": -0.3, "BEARISH": 0.7, "NEUTRAL": -0.2},
            "weakness": {"BULLISH": -0.35, "BEARISH": 0.65, "NEUTRAL": -0.15},
            "weak": {"BULLISH": -0.3, "BEARISH": 0.55, "NEUTRAL": -0.1},
            "worst": {"BULLISH": -0.35, "BEARISH": 0.75, "NEUTRAL": -0.2},
            "worse": {"BULLISH": -0.3, "BEARISH": 0.6, "NEUTRAL": -0.15},
            "deteriorate": {"BULLISH": -0.35, "BEARISH": 0.7, "NEUTRAL": -0.15},
            "collapse": {"BULLISH": -0.4, "BEARISH": 0.85, "NEUTRAL": -0.2},
            "fail": {"BULLISH": -0.35, "BEARISH": 0.7, "NEUTRAL": -0.15},
            "failed": {"BULLISH": -0.35, "BEARISH": 0.75, "NEUTRAL": -0.2},
            "scam": {"BULLISH": -0.4, "BEARISH": 0.8, "NEUTRAL": -0.2},
            "rekt": {"BULLISH": -0.35, "BEARISH": 0.85, "NEUTRAL": -0.25},
            "liquidated": {"BULLISH": -0.4, "BEARISH": 0.85, "NEUTRAL": -0.2},
            "bleed": {"BULLISH": -0.35, "BEARISH": 0.75, "NEUTRAL": -0.2},
            "blood": {"BULLISH": -0.3, "BEARISH": 0.7, "NEUTRAL": -0.2},
            "panic": {"BULLISH": -0.35, "BEARISH": 0.75, "NEUTRAL": -0.2},
            "fear": {"BULLISH": -0.3, "BEARISH": 0.65, "NEUTRAL": -0.15},
            
            # Neutral/uncertainty indicators
            "uncertain": {"BULLISH": -0.2, "BEARISH": -0.2, "NEUTRAL": 0.6},
            "uncertainty": {"BULLISH": -0.25, "BEARISH": -0.25, "NEUTRAL": 0.7},
            "possibly": {"BULLISH": -0.1, "BEARISH": -0.1, "NEUTRAL": 0.4},
            "maybe": {"BULLISH": -0.1, "BEARISH": -0.1, "NEUTRAL": 0.4},
            "might": {"BULLISH": -0.1, "BEARISH": -0.1, "NEUTRAL": 0.35},
            "could": {"BULLISH": -0.1, "BEARISH": -0.1, "NEUTRAL": 0.35},
            "volatile": {"BULLISH": -0.15, "BEARISH": -0.15, "NEUTRAL": 0.5},
            "volatility": {"BULLISH": -0.15, "BEARISH": -0.15, "NEUTRAL": 0.5},
            "mixed": {"BULLISH": -0.1, "BEARISH": -0.1, "NEUTRAL": 0.5},
            "sideways": {"BULLISH": -0.15, "BEARISH": -0.15, "NEUTRAL": 0.55},
            "consolidate": {"BULLISH": 0.0, "BEARISH": 0.0, "NEUTRAL": 0.5},
            "range": {"BULLISH": -0.1, "BEARISH": -0.1, "NEUTRAL": 0.45},
        }
        
        # Build vocabulary and weights
        for idx, (word, label_weights) in enumerate(features.items()):
            self._vocab[word] = idx
            for label, weight in label_weights.items():
                # INT8 quantization simulation (scale to -127 to 127)
                quant_weight = int(weight * 127) / 127.0
                self._weights[idx][label] = quant_weight
        
        self._trained = True
    
    def get_feature_indices(self, tokens: List[str]) -> List[int]:
        """Convert tokens to feature indices."""
        indices = []
        for token in tokens:
            if token in self._vocab:
                indices.append(self._vocab[token])
        return indices
    
    def predict(self, feature_indices: List[int]) -> Dict[str, float]:
        """
        Predict class scores given feature indices.
        Uses efficient sparse dot product.
        """
        scores = {label: self._biases[label] for label in self.LABELS}
        
        if not feature_indices:
            return scores
        
        # Sparse dot product
        for idx in feature_indices:
            if idx in self._weights:
                for label in self.LABELS:
                    scores[label] += self._weights[idx].get(label, 0.0)
        
        return scores


class FastTextClassifier:
    """
    Lightweight fastText-style classifier for news headlines.
    Optimized for microsecond inference with minimal RAM.
    """
    
    # Tokenization pattern
    WORD_PATTERN = re.compile(r'\b[a-zA-Z]{2,}\b')
    
    def __init__(self, model: Optional[QuantizedFastTextModel] = None):
        self.model = model or QuantizedFastTextModel()
        self._stopwords = self._load_stopwords()
    
    def _load_stopwords(self) -> set:
        """Load minimal stopword list."""
        return {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
            "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will", "would",
            "could", "should", "may", "might", "must", "shall", "can", "need",
            "this", "that", "these", "those", "it", "its", "as", "if", "when",
            "than", "because", "while", "although", "though", "after", "before",
            "until", "since", "where", "what", "who", "whom", "which", "whose",
            "why", "how", "all", "each", "every", "both", "few", "more", "most",
            "other", "some", "such", "no", "nor", "not", "only", "own", "same",
            "so", "just", "also", "now", "here", "there", "then", "once",
        }
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize and clean text, dropping stopwords."""
        words = self.WORD_PATTERN.findall(text.lower())
        return [w for w in words if w not in self._stopwords]
    
    def classify(self, text: str) -> ClassificationResult:
        """
        Classify text into BULLISH/BEARISH/NEUTRAL.
        Drops raw text immediately after feature extraction.
        """
        # Extract features
        tokens = self.tokenize(text)
        feature_indices = self.model.get_feature_indices(tokens)
        
        # Get raw scores
        raw_scores = self.model.predict(feature_indices)
        
        # Apply softmax normalization
        scores = self._softmax(raw_scores)
        
        # Determine prediction
        label = max(scores.keys(), key=lambda k: scores[k])
        confidence = scores[label]
        
        return ClassificationResult(
            label=label,
            confidence=confidence,
            scores=scores,
        )
    
    def _softmax(self, scores: Dict[str, float]) -> Dict[str, float]:
        """Apply softmax to convert scores to probabilities."""
        max_score = max(scores.values()) if scores else 0.0
        
        # Numerical stability
        exp_scores = {
            label: math.exp(score - max_score)
            for label, score in scores.items()
        }
        
        total = sum(exp_scores.values())
        if total == 0:
            # Uniform distribution if all scores are equal
            n_labels = len(scores)
            return {label: 1.0 / n_labels for label in scores}
        
        return {label: score / total for label, score in exp_scores.items()}
    
    def classify_batch(self, texts: List[str]) -> List[ClassificationResult]:
        """Classify multiple texts efficiently."""
        return [self.classify(text) for text in texts]
    
    def get_signal_strength(self, result: ClassificationResult) -> float:
        """
        Calculate signal strength for trading decisions.
        Returns value between -1 (strong bearish) and +1 (strong bullish).
        """
        if result.label == "BULLISH":
            return result.confidence
        elif result.label == "BEARISH":
            return -result.confidence
        else:
            return 0.0


def main():
    """Example usage of the fastText classifier."""
    classifier = FastTextClassifier()
    
    test_headlines = [
        "Bitcoin surges past $50k as institutional adoption grows",
        "Crypto market crashes amid regulatory crackdown fears",
        "Ethereum shows mixed signals as traders await Fed decision",
        "BTC rallies on strong earnings beat from major miners",
        "Altcoins tumble as Bitcoin dominance increases",
        "Market uncertainty persists ahead of CPI data release",
        "DeFi tokens soar following major protocol upgrade",
        "Exchange hacked, millions in crypto stolen",
        "Analysts optimistic about Q4 crypto performance",
        "Liquidations spike as leverage reaches extreme levels",
    ]
    
    print("fastText News Headline Classification:")
    print("=" * 70)
    
    for headline in test_headlines:
        result = classifier.classify(headline)
        signal = classifier.get_signal_strength(result)
        
        print(f"\nHeadline: {headline}")
        print(f"  Label: {result.label} (confidence: {result.confidence:.3f})")
        print(f"  Signal Strength: {signal:+.3f}")
        print(f"  Decisive: {result.is_decisive}")
        if result.is_decisive:
            print(f"  >>> TRADING SIGNAL: {'BUY' if signal > 0 else 'SELL'}")


if __name__ == "__main__":
    main()
