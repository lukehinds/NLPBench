"""
Diversity metrics for NLP datasets with tiered implementation approach.

This module provides comprehensive diversity measurements with two tiers:
1. Basic metrics: Using only built-in dependencies
2. Enhanced metrics: Using optional advanced NLP libraries
"""

import math
import re
from collections import Counter
from typing import Any

import numpy as np
import pandas as pd
from rich.console import Console

console = Console()


class DiversityMetrics:
    """Calculate various diversity metrics for text datasets with tiered functionality."""

    def __init__(self, enable_enhanced: bool = True):
        """
        Initialize diversity metrics calculator.

        Args:
            enable_enhanced: Whether to enable enhanced metrics (requires optional dependencies)
        """
        self.enable_enhanced = enable_enhanced
        self._enhanced_available = self._check_enhanced_dependencies()

        if enable_enhanced and not self._enhanced_available:
            console.print(
                "[yellow]Enhanced diversity metrics unavailable. Install with: pip install nlpbench[diversity][/yellow]"
            )
            self.enable_enhanced = False

    def _check_enhanced_dependencies(self) -> bool:
        """Check if enhanced dependencies are available."""
        import importlib.util

        required_packages = [
            "sentence_transformers",
            "sklearn",
            "nltk",
            "textstat",
            "spacy",
            "gensim",
        ]

        for package in required_packages:
            if importlib.util.find_spec(package) is None:
                return False

        return True

    def calculate_all_metrics(self, df: pd.DataFrame) -> dict[str, Any]:
        """
        Calculate all available diversity metrics.

        Args:
            df: DataFrame with 'content', 'role', and other columns

        Returns:
            dictionary containing all diversity metrics
        """
        metrics = {}

        # Basic diversity metrics (always available)
        try:
            metrics.update(self._calculate_basic_metrics(df))
        except Exception as e:
            console.print(f"[yellow]Warning: Basic metrics failed: {str(e)}[/yellow]")

        # Enhanced diversity metrics (if dependencies available)
        if self.enable_enhanced and self._enhanced_available:
            try:
                metrics.update(self._calculate_enhanced_metrics(df))
            except Exception as e:
                console.print(f"[yellow]Warning: Enhanced metrics failed: {str(e)}[/yellow]")

        return metrics

    def _calculate_basic_metrics(self, df: pd.DataFrame) -> dict[str, Any]:
        """Calculate basic diversity metrics using built-in dependencies."""

        content_series = df["content"].astype(str)
        metrics = {}

        # 1. Lexical Diversity (Basic)
        try:
            metrics["lexical_diversity"] = self._basic_lexical_diversity(content_series)
        except Exception as e:
            console.print(f"[yellow]Lexical diversity failed: {str(e)}[/yellow]")

        # 2. Length Diversity
        try:
            metrics["length_diversity"] = self._length_diversity(content_series)
        except Exception as e:
            console.print(f"[yellow]Length diversity failed: {str(e)}[/yellow]")

        # 3. Character Diversity
        try:
            metrics["character_diversity"] = self._character_diversity(content_series)
        except Exception as e:
            console.print(f"[yellow]Character diversity failed: {str(e)}[/yellow]")

        # 4. Word Frequency Diversity
        try:
            metrics["word_frequency_diversity"] = self._word_frequency_diversity(content_series)
        except Exception as e:
            console.print(f"[yellow]Word frequency diversity failed: {str(e)}[/yellow]")

        # 5. Role Distribution Diversity
        if "role" in df.columns:
            try:
                metrics["role_diversity"] = self._role_diversity(df["role"])
            except Exception as e:
                console.print(f"[yellow]Role diversity failed: {str(e)}[/yellow]")

        return metrics

    def _calculate_enhanced_metrics(self, df: pd.DataFrame) -> dict[str, Any]:
        """Calculate enhanced diversity metrics using optional dependencies."""

        content_series = df["content"].astype(str)
        metrics = {}

        # 6. Advanced Lexical Diversity
        metrics["advanced_lexical_diversity"] = self._advanced_lexical_diversity(content_series)

        # 7. Semantic Diversity
        metrics["semantic_diversity"] = self._semantic_diversity(content_series)

        # 8. Syntactic Diversity
        metrics["syntactic_diversity"] = self._syntactic_diversity(content_series)

        # 9. Topic Diversity
        metrics["topic_diversity"] = self._topic_diversity(content_series)

        # 10. Readability Diversity
        metrics["readability_diversity"] = self._readability_diversity(content_series)

        return metrics

    def _basic_lexical_diversity(self, content_series: pd.Series) -> dict[str, float]:
        """Calculate basic lexical diversity metrics."""

        # Combine all text
        all_text = " ".join(content_series.fillna(""))

        # Basic word tokenization
        words = re.findall(r"\b\w+\b", all_text.lower())

        if not words:
            return {
                "type_token_ratio": 0.0,
                "unique_word_ratio": 0.0,
                "vocabulary_size": 0,
                "total_words": 0,
            }

        unique_words = set(words)

        return {
            "type_token_ratio": len(unique_words) / len(words),
            "unique_word_ratio": len(unique_words) / len(set(words)) if words else 0.0,
            "vocabulary_size": len(unique_words),
            "total_words": len(words),
        }

    def _length_diversity(self, content_series: pd.Series) -> dict[str, float]:
        """Calculate content length diversity metrics."""

        lengths = content_series.str.len().fillna(0)

        if len(lengths) == 0:
            return {"length_variance": 0.0, "length_std": 0.0, "length_cv": 0.0}

        try:

            def safe_float_convert(val) -> float:
                """Safely convert pandas scalar to float."""
                try:
                    # Handle numpy scalar types
                    if hasattr(val, "item"):
                        return float(val.item())
                    # Handle regular numeric types
                    return float(val)
                except (ValueError, TypeError):
                    return 0.0

            variance = safe_float_convert(lengths.var())
            std = safe_float_convert(lengths.std())
            mean_length = safe_float_convert(lengths.mean())
            cv = std / mean_length if mean_length > 0 else 0.0
            length_range = safe_float_convert(lengths.max() - lengths.min())
            length_iqr = safe_float_convert(lengths.quantile(0.75) - lengths.quantile(0.25))
        except Exception as e:
            console.print(f"[yellow]Length diversity calculation error: {e}[/yellow]")
            return {
                "length_variance": 0.0,
                "length_std": 0.0,
                "length_cv": 0.0,
                "length_range": 0.0,
                "length_iqr": 0.0,
            }

        return {
            "length_variance": variance,
            "length_std": std,
            "length_cv": cv,  # Coefficient of variation
            "length_range": length_range,
            "length_iqr": length_iqr,
        }

    def _character_diversity(self, content_series: pd.Series) -> dict[str, float]:
        """Calculate character-level diversity."""

        # Combine all text
        all_text = "".join(content_series.fillna(""))

        if not all_text:
            return {"char_entropy": 0.0, "unique_char_ratio": 0.0}

        # Character frequency
        char_counts = Counter(all_text)
        total_chars = len(all_text)

        # Calculate entropy
        entropy = -sum(
            (count / total_chars) * math.log2(count / total_chars) for count in char_counts.values()
        )

        return {
            "char_entropy": entropy,
            "unique_char_ratio": len(char_counts) / total_chars,
            "unique_chars": len(char_counts),
        }

    def _word_frequency_diversity(self, content_series: pd.Series) -> dict[str, float]:
        """Calculate word frequency distribution diversity."""

        # Combine all text and tokenize
        all_text = " ".join(content_series.fillna(""))
        words = re.findall(r"\b\w+\b", all_text.lower())

        if not words:
            return {"word_entropy": 0.0, "hapax_ratio": 0.0}

        word_counts = Counter(words)
        total_words = len(words)

        # Calculate word entropy
        entropy = -sum(
            (count / total_words) * math.log2(count / total_words) for count in word_counts.values()
        )

        # Hapax legomena (words appearing only once)
        hapax_count = sum(1 for count in word_counts.values() if count == 1)

        return {
            "word_entropy": entropy,
            "hapax_ratio": hapax_count / len(word_counts) if word_counts else 0.0,
            "word_freq_variance": float(np.var(list(word_counts.values()))),
        }

    def _role_diversity(self, role_series: pd.Series) -> dict[str, float]:
        """Calculate role distribution diversity."""

        role_counts = role_series.value_counts()

        if len(role_counts) <= 1:
            total = len(role_series)
            return {
                "role_entropy": 0.0,
                "role_balance": 0.0,
                "unique_roles": len(role_counts),
                "dominant_role_ratio": 1.0 if total > 0 else 0.0,
            }

        total = len(role_series)

        # Role entropy
        entropy = -sum((count / total) * math.log2(count / total) for count in role_counts.values)

        # Role balance (closer to 1 = more balanced)
        max_possible_entropy = math.log2(len(role_counts))
        balance = entropy / max_possible_entropy if max_possible_entropy > 0 else 0.0

        return {
            "role_entropy": entropy,
            "role_balance": balance,
            "unique_roles": len(role_counts),
            "dominant_role_ratio": float(role_counts.max() / total),
        }

    def _advanced_lexical_diversity(self, content_series: pd.Series) -> dict[str, float]:
        """Calculate advanced lexical diversity using NLTK."""
        try:
            import nltk

            # Ensure required NLTK data is available
            try:
                nltk.data.find("tokenizers/punkt")
            except LookupError:
                nltk.download("punkt", quiet=True)

            metrics = {}
            all_texts = content_series.fillna("").tolist()

            # MTLD (Measure of Textual Lexical Diversity)
            metrics["mtld"] = self._calculate_mtld(all_texts)

            # Moving Average TTR
            metrics["mattr"] = self._calculate_mattr(all_texts)

            return metrics

        except Exception as e:
            console.print(f"[yellow]Advanced lexical diversity failed: {str(e)}[/yellow]")
            return {}

    def _semantic_diversity(self, content_series: pd.Series) -> dict[str, float]:
        """Calculate semantic diversity using embeddings."""
        try:
            import numpy as np
            from sentence_transformers import SentenceTransformer
            from sklearn.cluster import KMeans
            from sklearn.metrics.pairwise import cosine_similarity

            # Load lightweight model
            model = SentenceTransformer("all-MiniLM-L6-v2")

            texts = content_series.fillna("").tolist()[:100]  # Limit for performance

            if len(texts) < 2:
                return {"semantic_diversity_score": 0.0}

            # Generate embeddings
            embeddings = model.encode(texts, show_progress_bar=False)

            # Calculate pairwise similarities
            similarities = cosine_similarity(embeddings)

            # Remove diagonal (self-similarity)
            np.fill_diagonal(similarities, 0)

            # Semantic diversity = 1 - average similarity
            avg_similarity = np.mean(similarities[np.triu_indices_from(similarities, k=1)])

            # Cluster analysis
            if len(texts) >= 5:
                n_clusters = min(5, len(texts) // 2)
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                clusters = kmeans.fit_predict(embeddings)
                cluster_diversity = len(set(clusters)) / len(texts)
            else:
                cluster_diversity = 1.0

            return {
                "semantic_diversity_score": float(1 - avg_similarity),
                "avg_semantic_similarity": float(avg_similarity),
                "cluster_diversity": float(cluster_diversity),
            }

        except Exception as e:
            console.print(f"[yellow]Semantic diversity failed: {str(e)}[/yellow]")
            return {}

    def _syntactic_diversity(self, content_series: pd.Series) -> dict[str, float]:
        """Calculate syntactic diversity using spaCy."""
        try:
            import spacy

            # Try to load English model
            try:
                nlp = spacy.load("en_core_web_sm")
            except OSError:
                console.print(
                    "[yellow]spaCy English model not found. Run: python -m spacy download en_core_web_sm[/yellow]"
                )
                return {}

            texts = content_series.fillna("").tolist()[:50]  # Limit for performance

            pos_patterns = []
            dep_patterns = []

            for text in texts:
                if len(text.strip()) == 0:
                    continue

                doc = nlp(text[:500])  # Limit text length

                # POS patterns
                pos_sequence = [token.pos_ for token in doc if not token.is_space]
                if pos_sequence:
                    pos_patterns.append("_".join(pos_sequence[:10]))  # First 10 POS tags

                # Dependency patterns
                dep_sequence = [token.dep_ for token in doc if not token.is_space]
                if dep_sequence:
                    dep_patterns.append("_".join(dep_sequence[:10]))  # First 10 dependencies

            # Calculate pattern diversity
            unique_pos = len(set(pos_patterns))
            unique_dep = len(set(dep_patterns))

            pos_diversity = unique_pos / len(pos_patterns) if pos_patterns else 0.0
            dep_diversity = unique_dep / len(dep_patterns) if dep_patterns else 0.0

            return {
                "pos_pattern_diversity": pos_diversity,
                "dependency_pattern_diversity": dep_diversity,
                "unique_pos_patterns": unique_pos,
                "unique_dep_patterns": unique_dep,
            }

        except Exception as e:
            console.print(f"[yellow]Syntactic diversity failed: {str(e)}[/yellow]")
            return {}

    def _topic_diversity(self, content_series: pd.Series) -> dict[str, float]:
        """Calculate topic diversity using LDA."""
        try:
            from gensim import corpora
            from gensim.models import LdaModel

            texts = content_series.fillna("").tolist()

            # Simple preprocessing
            processed_texts = []
            for text in texts:
                # Basic cleaning and tokenization
                words = re.findall(r"\b\w+\b", text.lower())
                # Remove very short words and common stop words
                words = [
                    w
                    for w in words
                    if len(w) > 2
                    and w
                    not in {
                        "the",
                        "and",
                        "or",
                        "but",
                        "in",
                        "on",
                        "at",
                        "to",
                        "for",
                        "of",
                        "with",
                        "by",
                    }
                ]
                if words:
                    processed_texts.append(words)

            if len(processed_texts) < 2:
                return {"topic_diversity": 0.0}

            # Create dictionary and corpus
            dictionary = corpora.Dictionary(processed_texts)
            dictionary.filter_extremes(no_below=2, no_above=0.8)

            if len(dictionary) < 2:
                return {"topic_diversity": 0.0}

            corpus = [dictionary.doc2bow(text) for text in processed_texts]

            # Train LDA model
            num_topics = min(5, len(processed_texts) // 2, len(dictionary) // 2)
            if num_topics < 2:
                return {"topic_diversity": 0.0}

            lda_model = LdaModel(
                corpus=corpus,
                id2word=dictionary,
                num_topics=num_topics,
                random_state=42,
                passes=10,
                alpha="auto",
                per_word_topics=True,
                minimum_probability=0.0,
            )

            # Calculate topic diversity
            topic_distributions = []
            for doc in corpus:
                doc_topics = lda_model.get_document_topics(doc, minimum_probability=0.0)
                topic_probs = [prob for _, prob in doc_topics]
                topic_distributions.append(topic_probs)

            if not topic_distributions:
                return {"topic_diversity": 0.0}

            # Calculate average entropy of topic distributions
            topic_entropies = []
            for dist in topic_distributions:
                if sum(dist) > 0:
                    entropy = -sum(p * math.log2(p) for p in dist if p > 0)
                    topic_entropies.append(entropy)

            avg_topic_entropy = np.mean(topic_entropies) if topic_entropies else 0.0
            max_entropy = math.log2(num_topics)
            normalized_entropy = avg_topic_entropy / max_entropy if max_entropy > 0 else 0.0

            return {
                "topic_diversity": float(normalized_entropy),
                "num_topics_detected": num_topics,
                "avg_topic_entropy": float(avg_topic_entropy),
            }

        except Exception as e:
            console.print(f"[yellow]Topic diversity failed: {str(e)}[/yellow]")
            return {}

    def _readability_diversity(self, content_series: pd.Series) -> dict[str, float]:
        """Calculate readability diversity using textstat."""
        try:
            import textstat

            texts = content_series.fillna("").tolist()

            readability_scores = []
            for text in texts:
                if len(text.strip()) > 10:  # Only analyze substantial texts
                    # Use flesch reading ease method
                    score = textstat.flesch_reading_ease(text)  # type: ignore
                    readability_scores.append(score)

            if not readability_scores:
                return {"readability_diversity": 0.0}

            # Calculate variance in readability scores
            variance = float(np.var(readability_scores))
            std = float(np.std(readability_scores))
            mean_score = float(np.mean(readability_scores))

            return {
                "readability_diversity": variance,
                "readability_std": std,
                "avg_readability_score": mean_score,
                "readability_range": float(max(readability_scores) - min(readability_scores)),
            }

        except Exception as e:
            console.print(f"[yellow]Readability diversity failed: {str(e)}[/yellow]")
            return {}

    def _calculate_mtld(self, texts: list[str]) -> float:
        """Calculate Measure of Textual Lexical Diversity."""
        try:
            from nltk.tokenize import word_tokenize

            all_words = []
            for text in texts:
                tokens = word_tokenize(text.lower())
                all_words.extend([token for token in tokens if token.isalpha()])

            if len(all_words) < 50:  # MTLD needs sufficient tokens
                return 0.0

            # MTLD calculation with TTR threshold of 0.72
            def mtld_calc(words, threshold=0.72):
                if len(words) < 10:
                    return 0.0

                types = set()
                factors = 0
                start = 0

                for i, word in enumerate(words):
                    types.add(word)
                    ttr = len(types) / (i - start + 1)

                    if ttr <= threshold:
                        factors += 1
                        types = set()
                        start = i + 1

                # Handle remaining words
                if len(words) > start:
                    remaining_length = len(words) - start
                    if remaining_length > 0:
                        remaining_types = set(words[start:])
                        remaining_ttr = len(remaining_types) / remaining_length
                        factors += (1 - remaining_ttr) / (1 - threshold) if threshold < 1 else 0

                return len(all_words) / factors if factors > 0 else len(all_words)

            return mtld_calc(all_words)

        except Exception:
            return 0.0

    def _calculate_mattr(self, texts: list[str], window_size: int = 100) -> float:
        """Calculate Moving Average Type-Token Ratio."""
        try:
            from nltk.tokenize import word_tokenize

            all_words = []
            for text in texts:
                tokens = word_tokenize(text.lower())
                all_words.extend([token for token in tokens if token.isalpha()])

            if len(all_words) < window_size:
                unique_words = set(all_words)
                return len(unique_words) / len(all_words) if all_words else 0.0

            ttrs = []
            for i in range(len(all_words) - window_size + 1):
                window = all_words[i : i + window_size]
                unique_in_window = set(window)
                ttr = len(unique_in_window) / window_size
                ttrs.append(ttr)

            return float(np.mean(ttrs)) if ttrs else 0.0

        except Exception:
            return 0.0

    def calculate_diversity_score(self, metrics: dict[str, Any]) -> float:
        """
        Calculate an overall diversity score from individual metrics.

        Args:
            metrics: dictionary of calculated diversity metrics

        Returns:
            Overall diversity score between 0 and 100
        """
        scores = []
        weights = {}

        # Basic metrics (always available)
        if "lexical_diversity" in metrics:
            lex_metrics = metrics["lexical_diversity"]
            if "type_token_ratio" in lex_metrics:
                scores.append(min(lex_metrics["type_token_ratio"] * 100, 100))
                weights["lexical"] = 20

        if "character_diversity" in metrics:
            char_metrics = metrics["character_diversity"]
            if "char_entropy" in char_metrics:
                # Normalize entropy (typical max around 4.5 for English)
                normalized_entropy = min(char_metrics["char_entropy"] / 4.5 * 100, 100)
                scores.append(normalized_entropy)
                weights["character"] = 10

        if "word_frequency_diversity" in metrics:
            word_metrics = metrics["word_frequency_diversity"]
            if "word_entropy" in word_metrics:
                # Normalize word entropy
                normalized_entropy = min(word_metrics["word_entropy"] / 10 * 100, 100)
                scores.append(normalized_entropy)
                weights["word_frequency"] = 15

        if "role_diversity" in metrics:
            role_metrics = metrics["role_diversity"]
            if "role_balance" in role_metrics:
                scores.append(role_metrics["role_balance"] * 100)
                weights["role"] = 15

        # Enhanced metrics (if available)
        if "semantic_diversity" in metrics:
            sem_metrics = metrics["semantic_diversity"]
            if "semantic_diversity_score" in sem_metrics:
                scores.append(sem_metrics["semantic_diversity_score"] * 100)
                weights["semantic"] = 25

        if "syntactic_diversity" in metrics:
            syn_metrics = metrics["syntactic_diversity"]
            if "pos_pattern_diversity" in syn_metrics:
                scores.append(syn_metrics["pos_pattern_diversity"] * 100)
                weights["syntactic"] = 10

        if "topic_diversity" in metrics:
            topic_metrics = metrics["topic_diversity"]
            if "topic_diversity" in topic_metrics:
                scores.append(topic_metrics["topic_diversity"] * 100)
                weights["topic"] = 20

        # Calculate weighted average
        if not scores:
            return 0.0

        # Get corresponding weights for scores
        weight_values = list(weights.values())

        if len(weight_values) == 0 or sum(weight_values) == 0:
            return float(sum(scores) / len(scores))

        # Ensure we have matching number of scores and weights
        if len(scores) != len(weight_values):
            # Fallback to simple average
            return float(sum(scores) / len(scores))

        weighted_sum = sum(score * weight for score, weight in zip(scores, weight_values))
        total_weight = sum(weight_values)
        return min(weighted_sum / total_weight, 100.0)
