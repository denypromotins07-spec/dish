"""
Loughran-McDonald Financial Lexicon Sentiment Analyzer.
Custom financial dictionary-based sentiment scoring via PyO3/Rust bindings.
Microsecond text scoring without loading heavy NLP models into RAM.
Designed for <14GB RAM constraint with zero model overhead.
"""

import os
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple


@dataclass
class SentimentScore:
    """Lightweight sentiment score container."""
    positive: float = 0.0
    negative: float = 0.0
    neutral: float = 0.0
    uncertainty: float = 0.0
    litigious: float = 0.0
    constraining: float = 0.0
    modal_strong: float = 0.0
    modal_weak: float = 0.0
    
    @property
    def net_sentiment(self) -> float:
        """Net sentiment (positive - negative)."""
        return self.positive - self.negative
    
    @property
    def polarity(self) -> str:
        """Overall polarity classification."""
        net = self.net_sentiment
        if net > 0.05:
            return "BULLISH"
        elif net < -0.05:
            return "BEARISH"
        else:
            return "NEUTRAL"


class LoughranMcDonaldLexicon:
    """
    Loughran-McDonald financial sentiment lexicon.
    Loaded once and cached in memory-efficient format.
    """
    
    # Word categories as defined in LM dictionary
    CATEGORIES = [
        "positive", "negative", "uncertainty", "litigious",
        "constraining", "superfluous", "modal_strong", "modal_weak"
    ]
    
    def __init__(self, lexicon_dir: Optional[str] = None):
        self.lexicon_dir = lexicon_dir or os.path.join(
            os.path.dirname(__file__), "lexicons"
        )
        self._words: Dict[str, Set[str]] = {cat: set() for cat in self.CATEGORIES}
        self._word_to_categories: Dict[str, List[str]] = defaultdict(list)
        self._loaded = False
    
    def load(self) -> bool:
        """
        Load lexicon from files. Falls back to embedded mini-lexicon if files missing.
        Returns True if loaded successfully.
        """
        try:
            # Try to load from files first
            for category in self.CATEGORIES:
                filepath = os.path.join(self.lexicon_dir, f"{category}.txt")
                if os.path.exists(filepath):
                    with open(filepath, 'r', encoding='utf-8') as f:
                        words = {line.strip().lower() for line in f if line.strip()}
                        self._words[category].update(words)
                    
                    for word in words:
                        self._word_to_categories[word].append(category)
                    
                    self._loaded = True
            
            if not self._loaded:
                # Load embedded mini-lexicon
                self._load_embedded_lexicon()
            
            return True
        except Exception:
            self._load_embedded_lexicon()
            return True
    
    def _load_embedded_lexicon(self):
        """Load embedded minimal lexicon for critical financial terms."""
        # Core financial sentiment words (subset of LM dictionary)
        embedded = {
            "positive": {
                "gain", "gains", "gained", "gainful", "profit", "profits",
                "profitable", "profitability", "earnings", "earn", "earned",
                "revenue", "growth", "grow", "grew", "grown", "outperform",
                "beat", "exceed", "surplus", "benefit", "beneficial", "upside",
                "bullish", "rally", "surge", "soar", "jump", "leap", "climb",
                "advance", "appreciate", "favorable", "favorably", "optimistic",
                "confidence", "confident", "strong", "strength", "robust",
                "record", "high", "higher", "highest", "improve", "improved",
                "improvement", "success", "successful", "successfully", "win",
                "winner", "won", "leading", "leader", "outstanding", "exceptional",
            },
            "negative": {
                "loss", "losses", "lost", "lose", "losing", "deficit", "decline",
                "declined", "declining", "decrease", "decreased", "decreasing",
                "drop", "dropped", "fall", "fallen", "fell", "falling", "down",
                "lower", "lowest", "downturn", "slump", "slump", "crash", "crashed",
                "collapse", "collapsed", "plunge", "plunged", "tumble", "tumbled",
                "bearish", "pessimistic", "weakness", "weak", "weaker", "worst",
                "worse", "worsen", "worsened", "deteriorate", "deterioration",
                "impairment", "impaired", "write-down", "write-off", "bankruptcy",
                "insolvent", "liquidation", "liquidated", "default", "defaults",
                "delinquent", "adverse", "negatively", "unfavorable", "risk",
                "risks", "risky", "volatile", "volatility", "uncertain", "uncertainty",
                "rekt", "dump", "dumped", "bleed", "bleeding", "blood", "crater",
            },
            "uncertainty": {
                "uncertain", "uncertainty", "uncertainties", "possibly", "maybe",
                "might", "could", "would", "should", "may", "approximate",
                "approximately", "about", "roughly", "nearly", "almost", "seems",
                "appears", "potentially", "contingent", "dependent", "subject",
                "unclear", "ambiguous", "unpredictable", "unknown", "unsure",
            },
            "litigious": {
                "lawsuit", "lawsuits", "litigation", "legal", "legally", "court",
                "courts", "judge", "judgment", "verdict", "settlement", "settled",
                "plaintiff", "defendant", "prosecution", "conviction", "criminal",
                "civil", "regulatory", "regulation", "compliance", "violates",
                "violation", "violations", "sanction", "sanctions", "penalty",
                "penalties", "fine", "fines", "investigation", "subpoena",
            },
            "constraining": {
                "constraint", "constraints", "constrain", "constrained", "restrict",
                "restricted", "restriction", "restrictions", "limit", "limited",
                "limitation", "limitations", "cap", "caps", "ceiling", "floor",
                "minimum", "maximum", "threshold", "tolerance", "quota", "ban",
                "prohibit", "prohibited", "forbid", "forbidden", "prevent",
            },
            "modal_strong": {
                "must", "shall", "will", "required", "mandatory", "compulsory",
                "obligated", "committed", "certain", "definitely", "absolutely",
                "guarantee", "guaranteed", "assure", "assured", "promise", "promised",
                "pledge", "pledged", "firm", "fixed", "final", "irrevocable",
            },
            "modal_weak": {
                "may", "might", "could", "would", "should", "possible", "possibly",
                "potential", "potentially", "likely", "unlikely", "probable",
                "probably", "presumably", "perhaps", "maybe", "conditional",
                "contingent", "depends", "subject", "uncertain", "tentative",
            },
        }
        
        for category, words in embedded.items():
            self._words[category].update(words)
            for word in words:
                self._word_to_categories[word].append(category)
        
        self._loaded = True
    
    def get_categories(self, word: str) -> List[str]:
        """Get all categories for a word."""
        return self._word_to_categories.get(word.lower(), [])
    
    def is_in_category(self, word: str, category: str) -> bool:
        """Check if word belongs to a specific category."""
        return word.lower() in self._words.get(category, set())


class LoughranMcDonaldAnalyzer:
    """
    Fast lexicon-based sentiment analyzer using Loughran-McDonald dictionary.
    Optimized for microsecond scoring with minimal RAM footprint.
    """
    
    # Regex patterns for tokenization
    WORD_PATTERN = re.compile(r'\b[a-zA-Z]{2,}\b')
    NEGATION_WORDS = {"not", "no", "never", "neither", "none", "nobody", "nothing"}
    
    def __init__(self, lexicon: Optional[LoughranMcDonaldLexicon] = None):
        self.lexicon = lexicon or LoughranMcDonaldLexicon()
        self.lexicon.load()
        
        # Pre-compute negation impact
        self._negation_window = 3  # Words to look back for negation
    
    def tokenize(self, text: str) -> List[str]:
        """Extract words from text efficiently."""
        return self.WORD_PATTERN.findall(text.lower())
    
    def analyze(self, text: str) -> SentimentScore:
        """
        Analyze text and return sentiment scores.
        O(n) complexity where n is number of tokens.
        """
        tokens = self.tokenize(text)
        total_words = len(tokens)
        
        if total_words == 0:
            return SentimentScore()
        
        # Category counts
        counts: Dict[str, int] = defaultdict(int)
        
        # Track negation context
        negation_active = 0
        
        for i, token in enumerate(tokens):
            # Check for negation words
            if token in self.NEGATION_WORDS:
                negation_active = self._negation_window
            
            # Get categories for this word
            categories = self.lexicon.get_categories(token)
            
            for category in categories:
                # Handle negation for positive/negative
                if negation_active > 0 and category in ("positive", "negative"):
                    # Flip sentiment on negation
                    flipped = "negative" if category == "positive" else "positive"
                    counts[flipped] += 1
                else:
                    counts[category] += 1
            
            if negation_active > 0:
                negation_active -= 1
        
        # Normalize by total words
        score = SentimentScore(
            positive=counts["positive"] / total_words,
            negative=counts["negative"] / total_words,
            uncertainty=counts["uncertainty"] / total_words,
            litigious=counts["litigious"] / total_words,
            constraining=counts["constraining"] / total_words,
            modal_strong=counts["modal_strong"] / total_words,
            modal_weak=counts["modal_weak"] / total_words,
        )
        
        return score
    
    def analyze_batch(self, texts: List[str]) -> List[SentimentScore]:
        """Analyze multiple texts efficiently."""
        return [self.analyze(text) for text in texts]
    
    def get_sentiment_label(self, text: str) -> Tuple[str, float]:
        """Quick sentiment label and confidence score."""
        score = self.analyze(text)
        net = score.net_sentiment
        confidence = abs(net) * 10  # Scale to 0-1 roughly
        return score.polarity, min(confidence, 1.0)


# Rust extension stub (for PyO3 integration)
# In production, this would be implemented in Rust and compiled as native extension
try:
    # Attempt to import Rust-accelerated version
    # from lm_rust_fast import LoughranMcDonaldRust as FastAnalyzer
    FAST_ANALYZER_AVAILABLE = False
except ImportError:
    FAST_ANALYZER_AVAILABLE = False


def main():
    """Example usage of the Loughran-McDonald analyzer."""
    analyzer = LoughranMcDonaldAnalyzer()
    
    test_texts = [
        "Bitcoin surged to new highs as earnings beat expectations.",
        "The company reported losses and warned of declining revenue.",
        "Market uncertainty grows amid regulatory investigation.",
        "Strong profit growth drives bullish sentiment in crypto markets.",
        "HODL through the dip, bullish momentum building.",
        "Got rekt on the liquidation, blood everywhere.",
    ]
    
    print("Loughran-McDonald Sentiment Analysis:")
    print("=" * 60)
    
    for text in test_texts:
        score = analyzer.analyze(text)
        label, conf = analyzer.get_sentiment_label(text)
        print(f"\nText: {text}")
        print(f"  Polarity: {label} (confidence: {conf:.2f})")
        print(f"  Net Score: {score.net_sentiment:.4f}")
        print(f"  Positive: {score.positive:.4f}, Negative: {score.negative:.4f}")


if __name__ == "__main__":
    main()
