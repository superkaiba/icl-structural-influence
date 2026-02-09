"""
Natural Language Data Loader for Long Context Experiments.

This module provides utilities for loading natural language text from various
sources for long context collapse experiments:
- Wikipedia articles (WikiText-103)
- Books (Project Gutenberg via gutenberg library or direct download)
- Conversations (synthetic or from existing datasets)

These serve as comparison baselines against structured graph walks,
testing whether collapse is specific to synthetic data or a general phenomenon.

Usage:
    from src.data.natural_language_loader import NaturalLanguageLoader

    loader = NaturalLanguageLoader(tokenizer)
    tokens = loader.load_wikipedia(target_length=128000)
    tokens = loader.load_book(target_length=128000)
    tokens = loader.load_conversation(target_length=128000)
"""

import os
import random
import warnings
from pathlib import Path
from typing import Optional, Any
from dataclasses import dataclass


@dataclass
class NaturalLanguageConfig:
    """Configuration for natural language loading."""

    # Sources to use
    use_cached: bool = True  # Try to use cached data first
    cache_dir: str = ".cache/natural_language"

    # Wikipedia settings
    wiki_dataset: str = "wikitext-103-raw-v1"  # WikiText-103 variant

    # Book settings (Project Gutenberg IDs for classic public domain books)
    gutenberg_ids: list[int] = None

    # Conversation settings
    conversation_format: str = "user_assistant"  # or "multi_party"

    # Reproducibility
    seed: Optional[int] = 42

    def __post_init__(self):
        if self.gutenberg_ids is None:
            # Default to some classic novels with enough text
            self.gutenberg_ids = [
                1342,   # Pride and Prejudice
                11,     # Alice in Wonderland
                1661,   # Sherlock Holmes
                74,     # Tom Sawyer
                84,     # Frankenstein
                2701,   # Moby Dick
                98,     # A Tale of Two Cities
                1952,   # The Yellow Wallpaper
                174,    # Dorian Gray
                1400,   # Great Expectations
            ]


class NaturalLanguageLoader:
    """
    Loader for natural language text from various sources.

    Provides tokenized text for long context experiments, with options for
    Wikipedia, books, and conversations.
    """

    def __init__(
        self,
        tokenizer: Any,
        config: Optional[NaturalLanguageConfig] = None,
    ):
        """
        Initialize the loader.

        Args:
            tokenizer: HuggingFace tokenizer for encoding text
            config: Configuration options
        """
        self.tokenizer = tokenizer
        self.config = config or NaturalLanguageConfig()
        self.rng = random.Random(self.config.seed)

        # Ensure cache directory exists
        Path(self.config.cache_dir).mkdir(parents=True, exist_ok=True)

    def load_wikipedia(
        self,
        target_length: int,
        add_special_tokens: bool = False,
    ) -> list[int]:
        """
        Load Wikipedia text up to target token length.

        Uses WikiText-103 from HuggingFace datasets.

        Args:
            target_length: Target number of tokens
            add_special_tokens: Whether to add model's special tokens

        Returns:
            List of token IDs
        """
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("Please install datasets: pip install datasets")

        # Load WikiText-103
        cache_path = Path(self.config.cache_dir) / "wikitext_cache.txt"

        if self.config.use_cached and cache_path.exists():
            with open(cache_path, 'r', encoding='utf-8') as f:
                text = f.read()
        else:
            print("Loading WikiText-103 dataset...")
            ds = load_dataset("wikitext", self.config.wiki_dataset, split="train")

            # Concatenate articles (filter empty lines and headers)
            texts = []
            for item in ds:
                text = item["text"].strip()
                # Skip empty lines and section headers
                if text and not text.startswith("="):
                    texts.append(text)

            text = "\n\n".join(texts)

            # Cache for future use
            if self.config.use_cached:
                with open(cache_path, 'w', encoding='utf-8') as f:
                    f.write(text[:50_000_000])  # Cache ~50MB

        # Tokenize
        tokens = self.tokenizer.encode(
            text,
            add_special_tokens=add_special_tokens,
            truncation=False,
        )

        # Truncate to target length
        if len(tokens) > target_length:
            # Start from a random position for variety
            max_start = len(tokens) - target_length
            start = self.rng.randint(0, max_start) if max_start > 0 else 0
            tokens = tokens[start:start + target_length]

        return tokens

    def load_book(
        self,
        target_length: int,
        gutenberg_id: Optional[int] = None,
        add_special_tokens: bool = False,
    ) -> list[int]:
        """
        Load book text from Project Gutenberg.

        Args:
            target_length: Target number of tokens
            gutenberg_id: Specific Gutenberg book ID (None = random from list)
            add_special_tokens: Whether to add model's special tokens

        Returns:
            List of token IDs
        """
        if gutenberg_id is None:
            gutenberg_id = self.rng.choice(self.config.gutenberg_ids)

        cache_path = Path(self.config.cache_dir) / f"gutenberg_{gutenberg_id}.txt"

        if self.config.use_cached and cache_path.exists():
            with open(cache_path, 'r', encoding='utf-8') as f:
                text = f.read()
        else:
            text = self._download_gutenberg_book(gutenberg_id)
            if text and self.config.use_cached:
                with open(cache_path, 'w', encoding='utf-8') as f:
                    f.write(text)

        if not text:
            warnings.warn(f"Could not load book {gutenberg_id}, falling back to Wikipedia")
            return self.load_wikipedia(target_length, add_special_tokens)

        # Tokenize
        tokens = self.tokenizer.encode(
            text,
            add_special_tokens=add_special_tokens,
            truncation=False,
        )

        # If book is shorter than target, concatenate multiple books
        while len(tokens) < target_length:
            # Get another book
            next_id = self.rng.choice(self.config.gutenberg_ids)
            if next_id == gutenberg_id:
                continue

            next_text = self._load_or_download_gutenberg(next_id)
            if next_text:
                next_tokens = self.tokenizer.encode(
                    "\n\n---\n\n" + next_text,  # Separator between books
                    add_special_tokens=False,
                    truncation=False,
                )
                tokens.extend(next_tokens)

            gutenberg_id = next_id

        return tokens[:target_length]

    def _load_or_download_gutenberg(self, gutenberg_id: int) -> Optional[str]:
        """Load from cache or download a Gutenberg book."""
        cache_path = Path(self.config.cache_dir) / f"gutenberg_{gutenberg_id}.txt"

        if cache_path.exists():
            with open(cache_path, 'r', encoding='utf-8') as f:
                return f.read()

        text = self._download_gutenberg_book(gutenberg_id)
        if text:
            with open(cache_path, 'w', encoding='utf-8') as f:
                f.write(text)
        return text

    def _download_gutenberg_book(self, gutenberg_id: int) -> Optional[str]:
        """Download a book from Project Gutenberg."""
        import urllib.request
        import urllib.error

        # Try multiple mirror URLs
        urls = [
            f"https://www.gutenberg.org/cache/epub/{gutenberg_id}/pg{gutenberg_id}.txt",
            f"https://www.gutenberg.org/files/{gutenberg_id}/{gutenberg_id}-0.txt",
            f"https://www.gutenberg.org/files/{gutenberg_id}/{gutenberg_id}.txt",
        ]

        for url in urls:
            try:
                with urllib.request.urlopen(url, timeout=30) as response:
                    text = response.read().decode('utf-8', errors='ignore')

                # Strip Project Gutenberg header/footer
                text = self._strip_gutenberg_boilerplate(text)
                if len(text) > 1000:  # Sanity check
                    return text
            except (urllib.error.URLError, urllib.error.HTTPError) as e:
                continue

        warnings.warn(f"Could not download Gutenberg book {gutenberg_id}")
        return None

    def _strip_gutenberg_boilerplate(self, text: str) -> str:
        """Remove Project Gutenberg header and footer."""
        # Common start markers
        start_markers = [
            "*** START OF THIS PROJECT GUTENBERG",
            "*** START OF THE PROJECT GUTENBERG",
            "*END*THE SMALL PRINT",
        ]

        # Common end markers
        end_markers = [
            "*** END OF THIS PROJECT GUTENBERG",
            "*** END OF THE PROJECT GUTENBERG",
            "End of Project Gutenberg",
            "End of the Project Gutenberg",
        ]

        # Find start
        start_idx = 0
        for marker in start_markers:
            idx = text.find(marker)
            if idx != -1:
                # Move past the marker line
                newline_idx = text.find('\n', idx)
                if newline_idx != -1:
                    start_idx = newline_idx + 1
                break

        # Find end
        end_idx = len(text)
        for marker in end_markers:
            idx = text.find(marker)
            if idx != -1:
                end_idx = idx
                break

        return text[start_idx:end_idx].strip()

    def load_conversation(
        self,
        target_length: int,
        n_turns_per_conversation: int = 10,
        add_special_tokens: bool = False,
    ) -> list[int]:
        """
        Load or generate multi-turn conversation text.

        Generates synthetic user/assistant conversations using templates
        or loads from a conversation dataset.

        Args:
            target_length: Target number of tokens
            n_turns_per_conversation: Turns per conversation
            add_special_tokens: Whether to add model's special tokens

        Returns:
            List of token IDs
        """
        cache_path = Path(self.config.cache_dir) / "conversations_cache.txt"

        if self.config.use_cached and cache_path.exists():
            with open(cache_path, 'r', encoding='utf-8') as f:
                text = f.read()
        else:
            # Try to load from dataset first
            text = self._try_load_conversation_dataset()

            if not text:
                # Fall back to synthetic generation
                text = self._generate_synthetic_conversations(
                    target_tokens=target_length * 5,  # Generate more than needed
                    turns_per_conversation=n_turns_per_conversation,
                )

            if self.config.use_cached and text:
                with open(cache_path, 'w', encoding='utf-8') as f:
                    f.write(text[:10_000_000])  # Cache ~10MB

        if not text:
            warnings.warn("Could not load conversations, falling back to Wikipedia")
            return self.load_wikipedia(target_length, add_special_tokens)

        # Tokenize
        tokens = self.tokenizer.encode(
            text,
            add_special_tokens=add_special_tokens,
            truncation=False,
        )

        return tokens[:target_length]

    def _try_load_conversation_dataset(self) -> Optional[str]:
        """Try to load conversations from a HuggingFace dataset."""
        try:
            from datasets import load_dataset

            # Try OpenAssistant conversations
            ds = load_dataset("OpenAssistant/oasst1", split="train")

            conversations = []
            for item in ds:
                role = item.get("role", "user")
                text = item.get("text", "")
                if text:
                    prefix = "User:" if role == "prompter" else "Assistant:"
                    conversations.append(f"{prefix} {text}")

            if conversations:
                return "\n\n".join(conversations)

        except Exception as e:
            pass

        return None

    def load_wildchat_conversation(
        self,
        target_length: int,
        min_turns: int = 10,
        add_special_tokens: bool = False,
    ) -> list[int]:
        """
        Load long conversations from WildChat dataset.

        WildChat contains real user-ChatGPT conversations with many long exchanges.
        Filters for conversations with at least min_turns turns.

        Args:
            target_length: Target number of tokens
            min_turns: Minimum number of turns to consider a conversation "long"
            add_special_tokens: Whether to add model's special tokens

        Returns:
            List of token IDs
        """
        cache_path = Path(self.config.cache_dir) / f"wildchat_min{min_turns}_cache.txt"

        if self.config.use_cached and cache_path.exists():
            with open(cache_path, 'r', encoding='utf-8') as f:
                text = f.read()
        else:
            text = self._load_wildchat_long_conversations(min_turns=min_turns)

            if text and self.config.use_cached:
                with open(cache_path, 'w', encoding='utf-8') as f:
                    f.write(text[:20_000_000])  # Cache ~20MB

        if not text:
            warnings.warn("Could not load WildChat, falling back to synthetic conversations")
            return self.load_conversation(target_length, add_special_tokens=add_special_tokens)

        # Tokenize
        tokens = self.tokenizer.encode(
            text,
            add_special_tokens=add_special_tokens,
            truncation=False,
        )

        # Truncate or start from random position
        if len(tokens) > target_length:
            max_start = len(tokens) - target_length
            start = self.rng.randint(0, max_start) if max_start > 0 else 0
            tokens = tokens[start:start + target_length]

        return tokens[:target_length]

    def _load_wildchat_long_conversations(self, min_turns: int = 10) -> Optional[str]:
        """Load long conversations from WildChat dataset."""
        try:
            from datasets import load_dataset

            print(f"Loading WildChat dataset (filtering for conversations with >= {min_turns} turns)...")

            # Load with streaming to handle large dataset
            ds = load_dataset('allenai/WildChat', split='train', streaming=True)

            long_conversations = []
            conversation_count = 0
            total_checked = 0

            for item in ds:
                total_checked += 1

                # Check turn count
                turn_count = item.get('turn', 0)
                if turn_count < min_turns:
                    continue

                # Extract conversation
                conversation = item.get('conversation', [])
                if not conversation:
                    continue

                # Format conversation
                conv_parts = []
                for turn in conversation:
                    role = turn.get('role', 'user')
                    content = turn.get('content', '')
                    if content:
                        prefix = "User:" if role == "user" else "Assistant:"
                        conv_parts.append(f"{prefix} {content}")

                if conv_parts:
                    long_conversations.append("\n\n".join(conv_parts))
                    conversation_count += 1

                    if conversation_count % 100 == 0:
                        print(f"  Found {conversation_count} long conversations (checked {total_checked})...")

                # Collect enough conversations
                if conversation_count >= 1000:
                    break

                # Safety limit on checking
                if total_checked >= 500000:
                    break

            print(f"Loaded {conversation_count} long conversations from WildChat")

            if long_conversations:
                return "\n\n---\n\n".join(long_conversations)

        except Exception as e:
            warnings.warn(f"Error loading WildChat: {e}")

        return None

    def _generate_synthetic_conversations(
        self,
        target_tokens: int,
        turns_per_conversation: int = 10,
    ) -> str:
        """Generate synthetic conversations for testing."""

        # Conversation topics and templates
        topics = [
            "programming", "cooking", "travel", "science", "history",
            "music", "movies", "books", "sports", "technology",
            "philosophy", "art", "nature", "health", "education",
        ]

        user_templates = [
            "Can you tell me about {topic}?",
            "What do you know about {topic}?",
            "I'm interested in learning about {topic}. Can you help?",
            "How does {topic} work?",
            "What are the key concepts in {topic}?",
            "Can you explain {topic} in simple terms?",
            "What's the history of {topic}?",
            "What are common misconceptions about {topic}?",
            "How can I get started with {topic}?",
            "What resources do you recommend for learning {topic}?",
        ]

        assistant_templates = [
            "I'd be happy to help you with {topic}. {topic_content}",
            "Great question about {topic}! {topic_content}",
            "Let me explain {topic}. {topic_content}",
            "{topic_content} Does that help clarify {topic}?",
            "Here's what I know about {topic}: {topic_content}",
        ]

        topic_content = {
            "programming": "Programming involves writing instructions for computers to execute. It requires logical thinking and attention to detail.",
            "cooking": "Cooking is both an art and a science. Understanding basic techniques like sauteing, braising, and roasting opens up endless possibilities.",
            "travel": "Travel broadens perspectives and creates lasting memories. Planning ahead while leaving room for spontaneity often leads to the best experiences.",
            "science": "Science is the systematic study of the natural world through observation and experimentation. It helps us understand everything from atoms to galaxies.",
            "history": "History helps us understand how we got here and avoid repeating past mistakes. Every era offers lessons relevant to today.",
            "music": "Music is a universal language that transcends cultural boundaries. From classical to modern genres, it evokes emotions and brings people together.",
            "movies": "Cinema combines storytelling with visual art to create immersive experiences. Different genres serve different purposes, from entertainment to education.",
            "books": "Books are windows into different worlds and minds. Reading develops vocabulary, critical thinking, and empathy.",
            "sports": "Sports promote physical health, teamwork, and discipline. Whether playing or watching, they bring communities together.",
            "technology": "Technology shapes how we live and work. Understanding its evolution helps us adapt to and shape its future direction.",
            "philosophy": "Philosophy tackles fundamental questions about existence, knowledge, and ethics. It provides frameworks for thinking about complex issues.",
            "art": "Art expresses human creativity and emotion in visual form. From cave paintings to digital art, it reflects our evolving culture.",
            "nature": "Nature encompasses all life on Earth and its interconnected systems. Understanding ecology helps us protect our environment.",
            "health": "Health is multifaceted, involving physical, mental, and social well-being. Prevention and lifestyle choices play crucial roles.",
            "education": "Education empowers individuals and societies. Effective learning combines knowledge acquisition with critical thinking skills.",
        }

        conversations = []
        estimated_tokens = 0

        while estimated_tokens < target_tokens:
            topic = self.rng.choice(topics)
            conv_parts = []

            for turn in range(turns_per_conversation):
                # User turn
                user_msg = self.rng.choice(user_templates).format(topic=topic)
                conv_parts.append(f"User: {user_msg}")

                # Assistant turn
                content = topic_content.get(topic, f"Let me share my thoughts on {topic}.")
                assistant_msg = self.rng.choice(assistant_templates).format(
                    topic=topic, topic_content=content
                )
                conv_parts.append(f"Assistant: {assistant_msg}")

            conversation = "\n".join(conv_parts)
            conversations.append(conversation)

            # Rough token estimate (4 chars per token)
            estimated_tokens += len(conversation) // 4

        return "\n\n---\n\n".join(conversations)

    def get_source_metadata(self, source: str) -> dict:
        """Get metadata about a data source."""
        metadata = {
            "source": source,
            "loader_class": "NaturalLanguageLoader",
        }

        if source == "wikipedia":
            metadata["dataset"] = self.config.wiki_dataset
        elif source == "book":
            metadata["gutenberg_ids"] = self.config.gutenberg_ids
        elif source == "conversation":
            metadata["format"] = self.config.conversation_format

        return metadata


def demo():
    """Demonstrate natural language loader functionality."""
    print("=" * 60)
    print("Natural Language Loader Demo")
    print("=" * 60)

    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("Please install transformers: pip install transformers")
        return

    # Use GPT-2 tokenizer for demo
    print("\nLoading GPT-2 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    loader = NaturalLanguageLoader(tokenizer)

    # Test each source
    target_length = 1000

    print(f"\nTarget length: {target_length} tokens")
    print("-" * 60)

    # Wikipedia
    print("\n1. Wikipedia:")
    try:
        tokens = loader.load_wikipedia(target_length)
        text = tokenizer.decode(tokens[:100])
        print(f"   Loaded {len(tokens)} tokens")
        print(f"   Sample: {text[:80]}...")
    except Exception as e:
        print(f"   Error: {e}")

    # Conversations (synthetic)
    print("\n2. Conversations:")
    try:
        tokens = loader.load_conversation(target_length)
        text = tokenizer.decode(tokens[:100])
        print(f"   Loaded {len(tokens)} tokens")
        print(f"   Sample: {text[:80]}...")
    except Exception as e:
        print(f"   Error: {e}")

    # Books (may require network)
    print("\n3. Books (Project Gutenberg):")
    try:
        tokens = loader.load_book(target_length, gutenberg_id=11)  # Alice in Wonderland
        text = tokenizer.decode(tokens[:100])
        print(f"   Loaded {len(tokens)} tokens")
        print(f"   Sample: {text[:80]}...")
    except Exception as e:
        print(f"   Error: {e}")

    print("\n" + "=" * 60)
    print("Demo Complete")
    print("=" * 60)


if __name__ == "__main__":
    demo()
