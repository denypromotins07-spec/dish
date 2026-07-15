"""
VADER Sentiment Analyzer Adapted for Crypto Slang.
Extremely lightweight, CPU-efficient sentiment scoring tuned for crypto terminology.
No heavy NLP models - pure dictionary-based approach for <14GB RAM constraint.
"""

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class VaderCryptoScore:
    """VADER-style sentiment score with crypto-specific metrics."""
    compound: float = 0.0      # Overall sentiment (-1 to +1)
    positive: float = 0.0      # Positive proportion
    neutral: float = 0.0       # Neutral proportion
    negative: float = 0.0      # Negative proportion
    hype_score: float = 0.0    # Crypto-specific hype indicator
    fear_score: float = 0.0    # Fear/panic indicator
    
    @property
    def polarity(self) -> str:
        """Classify overall polarity."""
        if self.compound >= 0.05:
            return "BULLISH"
        elif self.compound <= -0.05:
            return "BEARISH"
        else:
            return "NEUTRAL"
    
    @property
    def intensity(self) -> str:
        """Classify sentiment intensity."""
        abs_compound = abs(self.compound)
        if abs_compound >= 0.75:
            return "EXTREME"
        elif abs_compound >= 0.5:
            return "STRONG"
        elif abs_compound >= 0.25:
            return "MODERATE"
        else:
            return "WEAK"


class VADERCryptoLexicon:
    """
    Crypto-adapted VADER lexicon with slang terms.
    Memory-efficient dictionary storage.
    """
    
    def __init__(self):
        self._word_scores: Dict[str, float] = {}
        self._punctuation_boost = {
            "!": 0.292,
            "!!": 0.358,
            "!!!": 0.438,
            "?": 0.18,
            "??": 0.25,
        }
        self._capitalization_boost = 0.733
        self._negation_words = {"not", "no", "never", "none", "neither", "nobody"}
        self._degree_modifiers = {
            "very": 1.25,
            "extremely": 1.5,
            "incredibly": 1.5,
            "super": 1.3,
            "mega": 1.4,
            "ultra": 1.4,
            "slightly": 0.5,
            "somewhat": 0.75,
            "barely": 0.5,
            "hardly": 0.5,
        }
        
        self._load_lexicon()
    
    def _load_lexicon(self):
        """Load crypto-adapted VADER lexicon."""
        # Standard VADER core words (subset)
        base_words = {
            # Positive words
            "good": 1.9, "great": 2.5, "excellent": 2.8, "amazing": 2.6,
            "awesome": 2.4, "fantastic": 2.7, "wonderful": 2.5, "love": 2.2,
            "like": 1.5, "happy": 2.1, "excited": 2.3, "profit": 2.0,
            "profits": 2.0, "gain": 1.8, "gains": 1.8, "bullish": 2.3,
            "moon": 2.5, "rocket": 2.4, "surge": 1.9, "rally": 2.0,
            "breakout": 2.1, "pump": 1.8, "green": 1.5, "up": 1.2,
            "higher": 1.4, "highest": 1.8, "win": 2.0, "winner": 2.2,
            "success": 2.1, "successful": 2.2, "beat": 1.7, "outperform": 2.0,
            "strong": 1.6, "strength": 1.7, "robust": 1.8, "optimistic": 2.0,
            "confidence": 1.8, "confident": 1.9, "hope": 1.5, "hopeful": 1.8,
            "buy": 1.3, "accumulate": 1.5, "hold": 0.8, "diamond": 1.6,
            "hands": 0.5, "hodl": 1.8, "tothemoon": 2.6, "lambo": 1.9,
            
            # Negative words
            "bad": -1.9, "terrible": -2.5, "awful": -2.4, "horrible": -2.6,
            "hate": -2.2, "dislike": -1.5, "sad": -1.8, "angry": -1.9,
            "loss": -2.0, "losses": -2.0, "lose": -1.9, "losing": -1.8,
            "lost": -1.9, "bearish": -2.3, "crash": -2.4, "dump": -2.0,
            "dumped": -2.1, "plunge": -2.2, "drop": -1.7, "fall": -1.6,
            "fallen": -1.7, "decline": -1.6, "declining": -1.5, "down": -1.2,
            "lower": -1.4, "lowest": -1.8, "red": -1.5, "sell": -1.3,
            "selling": -1.4, "panic": -2.0, "fear": -1.8, "worried": -1.6,
            "worry": -1.5, "anxious": -1.7, "stress": -1.6, "stressed": -1.7,
            "rekt": -2.5, "wrecked": -2.4, "liquidated": -2.3, "margin": -0.5,
            "call": -0.3, "bleed": -2.0, "bleeding": -2.1, "blood": -1.9,
            "crater": -2.2, "tank": -1.9, "tanked": -2.0, "collapse": -2.3,
            "fail": -2.0, "failed": -2.1, "failure": -2.2, "scam": -2.4,
            "rug": -2.5, "rugpull": -2.8, "exit": -0.8, "scam": -2.4,
            
            # Uncertainty words
            "uncertain": -1.2, "uncertainty": -1.3, "maybe": -0.5,
            "possibly": -0.4, "might": -0.3, "could": -0.2, "would": -0.1,
            "should": -0.1, "risk": -1.0, "risky": -1.2, "volatile": -0.8,
            "volatility": -0.7, "warning": -1.2, "warn": -1.1, "caution": -1.0,
        }
        
        # Crypto slang and community terms
        crypto_slang = {
            # Bullish slang
            "bull": 2.0, "bulls": 2.0, "bullrun": 2.4, "bullish": 2.3,
            "fomo": 1.5, "dyor": 0.8, "dd": 1.2, "gem": 1.8, "gems": 1.8,
            "100x": 2.5, "50x": 2.3, "10x": 2.0, "mooning": 2.4,
            "sendit": 1.6, "apes": 1.3, "ape": 1.2, "whale": 0.8,
            "whales": 0.9, "accumulation": 1.5, "accumulate": 1.5,
            "long": 1.2, "longs": 1.2, "squeeze": 1.4, "shorts": -0.5,
            "short": -0.8, "shorting": -1.0, "fud": -1.8, "shill": -1.2,
            
            # Bearish slang
            "bear": -2.0, "bears": -2.0, "bearish": -2.3, "bagholder": -1.8,
            "bags": -1.5, "paper": -1.2, "hands": -0.5, "weak": -1.3,
            "weakhands": -1.8, "jeets": -1.6, "jeet": -1.5, "chad": 1.5,
            "ngmi": -2.0, "gonewrong": -1.8, "copium": -1.2, "hopium": -0.8,
            "delayed": -1.0, "delay": -0.9, "halted": -1.5, "halt": -1.3,
            
            # Neutral/community
            "crypto": 0.1, "bitcoin": 0.2, "btc": 0.2, "ethereum": 0.2,
            "eth": 0.2, "altcoin": 0.1, "alts": 0.1, "defi": 0.3,
            "nft": 0.2, "web3": 0.3, "dao": 0.2, "staking": 0.5,
            "yield": 0.6, "farm": 0.4, "farming": 0.5, "airdrop": 1.0,
        }
        
        # Merge all lexicons
        self._word_scores.update(base_words)
        self._word_scores.update(crypto_slang)
    
    def get_score(self, word: str) -> float:
        """Get sentiment score for a word."""
        return self._word_scores.get(word.lower(), 0.0)
    
    def is_negation(self, word: str) -> bool:
        """Check if word is a negation."""
        return word.lower() in self._negation_words
    
    def get_degree_modifier(self, word: str) -> float:
        """Get degree modifier multiplier."""
        return self._degree_modifiers.get(word.lower(), 1.0)
    
    def get_punctuation_boost(self, punctuation: str) -> float:
        """Get punctuation boost value."""
        return self._punctuation_boost.get(punctuation, 0.0)


class VADERCryptoAnalyzer:
    """
    VADER-style sentiment analyzer optimized for crypto text.
    Extremely lightweight with no external dependencies.
    """
    
    # Tokenization patterns
    WORD_PATTERN = re.compile(r'\b[a-zA-Z]+\b')
    PUNCT_PATTERN = re.compile(r'[!?]+')
    
    def __init__(self, lexicon: Optional[VADERCryptoLexicon] = None):
        self.lexicon = lexicon or VADERCryptoLexicon()
        self._negation_window = 3
        self._degree_window = 2
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        return self.WORD_PATTERN.findall(text.lower())
    
    def count_punctuation(self, text: str) -> Dict[str, int]:
        """Count exclamation and question marks."""
        matches = self.PUNCT_PATTERN.findall(text)
        counts = defaultdict(int)
        for match in matches:
            counts[match] += 1
        return dict(counts)
    
    def has_capitalization_emphasis(self, text: str) -> bool:
        """Check if text has ALL CAPS words (emphasis)."""
        words = text.split()
        return any(w.isupper() and len(w) > 2 for w in words)
    
    def analyze(self, text: str) -> VaderCryptoScore:
        """
        Analyze text and return VADER-style scores.
        Optimized for speed with O(n) complexity.
        """
        tokens = self.tokenize(text)
        total_tokens = len(tokens)
        
        if total_tokens == 0:
            return VaderCryptoScore()
        
        # Track scores
        pos_sum = 0.0
        neg_sum = 0.0
        neu_sum = 0.0
        hype_terms = 0
        fear_terms = 0
        
        # Negation tracking
        negation_countdown = 0
        degree_multiplier = 1.0
        
        for i, token in enumerate(tokens):
            # Check for degree modifiers
            degree_mult = self.lexicon.get_degree_modifier(token)
            if degree_mult != 1.0:
                degree_multiplier = degree_mult
                continue
            
            # Check for negations
            if self.lexicon.is_negation(token):
                negation_countdown = self._negation_window
                degree_multiplier = 1.0
                neu_sum += 1
                continue
            
            # Get word score
            base_score = self.lexicon.get_score(token)
            
            # Apply negation
            if negation_countdown > 0:
                base_score = -base_score
                negation_countdown -= 1
            
            # Apply degree modifier
            if degree_multiplier != 1.0 and base_score != 0:
                base_score *= degree_multiplier
                degree_multiplier = 1.0
            
            # Categorize and sum
            if base_score > 0:
                pos_sum += base_score
            elif base_score < 0:
                neg_sum += abs(base_score)
            else:
                neu_sum += 1
            
            # Track crypto-specific metrics
            if token in {"moon", "rocket", "lambo", "100x", "10x", "gem", "gems"}:
                hype_terms += 1
            if token in {"rekt", "panic", "fear", "bleed", "crash", "dump", "liquidated"}:
                fear_terms += 1
        
        # Calculate punctuation boost
        punct_counts = self.count_punctuation(text)
        punct_boost = sum(
            self.lexicon.get_punctuation_boost(p) * count
            for p, count in punct_counts.items()
        )
        
        # Capitalization boost
        cap_boost = self._lexicon.capitalization_boost if self.has_capitalization_emphasis(text) else 0
        
        # Normalize scores
        total = pos_sum + neg_sum + neu_sum
        if total > 0:
            pos_ratio = pos_sum / total
            neg_ratio = neg_sum / total
            neu_ratio = neu_sum / total
        else:
            pos_ratio = neg_ratio = neu_ratio = 0.0
        
        # Compound score (normalized to -1, 1)
        raw_compound = pos_sum - neg_sum + punct_boost + cap_boost
        # Sigmoid-like normalization
        compound = raw_compound / (1.0 + abs(raw_compound))
        
        # Hype and fear scores
        hype_score = min(hype_terms / max(total_tokens, 1) * 5, 1.0)
        fear_score = min(fear_terms / max(total_tokens, 1) * 5, 1.0)
        
        return VaderCryptoScore(
            compound=compound,
            positive=pos_ratio,
            neutral=neu_ratio,
            negative=neg_ratio,
            hype_score=hype_score,
            fear_score=fear_score,
        )
    
    def analyze_batch(self, texts: List[str]) -> List[VaderCryptoScore]:
        """Analyze multiple texts efficiently."""
        return [self.analyze(text) for text in texts]
    
    def get_signal(self, text: str) -> Tuple[str, float]:
        """Get trading signal from text."""
        score = self.analyze(text)
        
        # Combine compound with hype/fear for signal
        if score.compound > 0.1 and score.hype_score > 0.3:
            return "STRONG_BUY", score.compound
        elif score.compound > 0.05:
            return "BUY", score.compound
        elif score.compound < -0.1 and score.fear_score > 0.3:
            return "STRONG_SELL", score.compound
        elif score.compound < -0.05:
            return "SELL", score.compound
        else:
            return "HOLD", score.compound


# Fix the reference to lexicon capitalization_boost
VADERCryptoAnalyzer._lexicon = VADERCryptoLexicon()


def main():
    """Example usage of VADER crypto analyzer."""
    analyzer = VADERCryptoAnalyzer()
    
    test_texts = [
        "Bitcoin is going to the moon! 🚀🚀🚀",
        "Got absolutely rekt on this dump, blood everywhere.",
        "HODL strong, diamond hands will be rewarded!",
        "FUD spreading, panic selling intensifies.",
        "Just accumulated more BTC, bullish AF!!!",
        "NGMI if you sell now, just wait for the next bull run.",
        "Whale alert! Someone is accumulating heavily.",
        "This is a rug pull, exit immediately!",
    ]
    
    print("VADER Crypto Sentiment Analysis:")
    print("=" * 70)
    
    for text in test_texts:
        score = analyzer.analyze(text)
        signal, confidence = analyzer.get_signal(text)
        
        print(f"\nText: {text}")
        print(f"  Signal: {signal}")
        print(f"  Compound: {score.compound:.3f} | Polarity: {score.polarity}")
        print(f"  Intensity: {score.intensity}")
        print(f"  Hype: {score.hype_score:.3f} | Fear: {score.fear_score:.3f}")


if __name__ == "__main__":
    main()
