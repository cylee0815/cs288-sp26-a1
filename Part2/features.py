from collections import ChainMap
from typing import Callable, Dict, Set

import re
import pandas as pd


class FeatureMap:
    name: str

    @classmethod
    def featurize(self, text: str) -> Dict[str, float]:
        pass

    @classmethod
    def prefix_with_name(self, d: Dict) -> Dict[str, float]:
        """just a handy shared util function"""
        return {f"{self.name}/{k}": v for k, v in d.items()}


class BagOfWords(FeatureMap):
    name = "bow"
    STOP_WORDS = set(pd.read_csv("stopwords.txt", header=None)[0])

    @classmethod
    def featurize(self, text: str) -> Dict[str, float]:
        # TODO: implement this! Expected # of lines: <5
        # raise NotImplementedError
        # return self.prefix_with_name({})
        # # 1. Normalize, split, and filter stop words
        # tokens = [w for w in text.lower().split() if w not in self.STOP_WORDS]
        # # 2. Count frequencies
        # counts = {token: float(tokens.count(token)) for token in set(tokens)}
        # return self.prefix_with_name(counts)
        # Fixed: It seems that we only wish to have a count of 1 for each unique word
        tokens = {w for w in text.lower().split() if w not in self.STOP_WORDS}
        return self.prefix_with_name({t: 1.0 for t in tokens})


class SentenceLength(FeatureMap):
    name = "len"

    @classmethod
    def featurize(self, text: str) -> Dict[str, float]:
        """an example of custom feature that rewards long sentences"""
        if len(text.split()) < 10:
            k = "short"
            v = 1.0
        else:
            k = "long"
            v = 5.0
        ret = {k: v}
        return self.prefix_with_name(ret)


class StyleFeatures(FeatureMap):
    name = "style"

    @classmethod
    def featurize(cls, text: str) -> Dict[str, float]:
        """Extracts stylistic counts like punctuation and capitalization."""
        return cls.prefix_with_name({
            "n_caps": float(sum(1 for c in text if c.isupper())),
            "n_excl": float(text.count("!")),
            "n_ques": float(text.count("?")),
            "n_digits": float(sum(1 for c in text if c.isdigit())),
            "n_words": float(len(text.split()))
        })


class BigramFeatures(FeatureMap):
    """Captures context by looking at pairs of words: 'not good', 'very happy'"""
    name = "bigram"

    @classmethod
    def featurize(cls, text: str) -> Dict[str, float]:
        words = text.lower().split()
        if len(words) < 2: 
            return {}
        # Zip creates pairs: (word1, word2), (word2, word3)...
        bigrams = {f"{w1}_{w2}" for w1, w2 in zip(words, words[1:])}
        return cls.prefix_with_name({b: 1.0 for b in bigrams})


class ComplexityFeatures(FeatureMap):
    """Calculates vocabulary complexity."""
    name = "complexity"

    @classmethod
    def featurize(cls, text: str) -> Dict[str, float]:
        words = text.split()
        if not words: return cls.prefix_with_name({"avg_word_len": 0.0})
        
        avg_len = sum(len(w) for w in words) / len(words)
        return cls.prefix_with_name({"avg_word_len": avg_len})


class RatioFeatures(FeatureMap):
    """
    NORMALIZED counts. 
    5 capitals in a 5 word sentence is intense. 
    5 capitals in a 1000 word essay is nothing.
    """
    name = "ratios"

    @classmethod
    def featurize(cls, text: str) -> Dict[str, float]:
        n_char = len(text) if len(text) > 0 else 1
        return cls.prefix_with_name({
            "caps_ratio": sum(1 for c in text if c.isupper()) / n_char,
            "excl_ratio": text.count("!") / n_char,
            "digit_ratio": sum(1 for c in text if c.isdigit()) / n_char
        })


class SentimentFeatures(FeatureMap):
    """
    Checks words against a known list of positive/negative words.
    This is usually the HIGHEST IMPACT feature for reviews.
    """
    name = "sentiment"
    
    # In a real app, load these from files (e.g., 'positive-words.txt')
    POSITIVE_WORDS = {"love", "good", "great", "amazing", "excellent", "best", "wonderful", "enjoy", "fantastic", "happy"}
    NEGATIVE_WORDS = {"hate", "bad", "terrible", "worst", "awful", "boring", "useless", "stupid", "waste", "poor"}

    @classmethod
    def featurize(cls, text: str) -> Dict[str, float]:
        tokens = text.lower().split()
        
        pos_count = sum(1 for t in tokens if t in cls.POSITIVE_WORDS)
        neg_count = sum(1 for t in tokens if t in cls.NEGATIVE_WORDS)
        
        # Calculate a "net score" (pos - neg)
        score = pos_count - neg_count
        
        return cls.prefix_with_name({
            "n_pos": float(pos_count),
            "n_neg": float(neg_count),
            "score": float(score)
        })


class EmoticonFeatures(FeatureMap):
    """
    Captures non-word sentiment signals like :) or :(
    """
    name = "emoji"

    @classmethod
    def featurize(cls, text: str) -> Dict[str, float]:
        # Simple regex for positive/negative smileys
        n_smile = len(re.findall(r'[:;=8]-?[\)D\]\}]', text))
        n_frown = len(re.findall(r'[:;=8]-?[\(\[\|\{]', text))
        
        return cls.prefix_with_name({
            "has_smile": 1.0 if n_smile > 0 else 0.0,
            "has_frown": 1.0 if n_frown > 0 else 0.0,
            "net_emoji": float(n_smile - n_frown)
        })


FEATURE_CLASSES_MAP = {
    c.name: c for c in [BagOfWords,
                        SentenceLength,
                        StyleFeatures,
                        BigramFeatures,
                        ComplexityFeatures,
                        RatioFeatures,
                        SentimentFeatures,
                        EmoticonFeatures]
}


def make_featurize(
    feature_types: Set[str],
) -> Callable[[str], Dict[str, float]]:
    featurize_fns = [FEATURE_CLASSES_MAP[n].featurize for n in feature_types]

    def _featurize(text: str):
        f = ChainMap(*[fn(text) for fn in featurize_fns])
        return dict(f)

    return _featurize


__all__ = ["make_featurize"]

if __name__ == "__main__":
    text = "I love this movie"
    print(text)
    print(BagOfWords.featurize(text))
    featurize = make_featurize({"bow", "len"})
    print(featurize(text))
    print()
    # custom case
    text = "I love this movie!!! It is 10/10."
    print(f"Analyzing: '{text}'\n")

    # Test the new feature specifically
    print("--- Style Features ---")
    print(StyleFeatures.featurize(text))
    
    print("\n--- Combined Features ---")
    # Add "style" to the requested feature set
    featurize = make_featurize({"bow", "len", "style"})
    
    # Pretty print the result
    result = featurize(text)
    for key, value in result.items():
        print(f"{key}: {value}")
