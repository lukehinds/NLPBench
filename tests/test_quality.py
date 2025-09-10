import pandas as pd
import pytest

from src.models import ConversationEntry, DatasetConfig
from src.quality_checker import DatasetQualityChecker


class TestQualityChecker:
    def test_conversation_entry_validation(self):
        """Test ConversationEntry validation."""

        # Valid entry
        entry = ConversationEntry(role="user", content="Hello world!")
        assert entry.role == "user"
        assert entry.content == "Hello world!"

        # Test role normalization
        entry = ConversationEntry(role=" USER ", content="Hello world!")
        assert entry.role == "user"

        # Test empty validation
        with pytest.raises(ValueError):
            ConversationEntry(role="", content="Hello world!")

        with pytest.raises(ValueError):
            ConversationEntry(role="user", content="")

    def test_dataset_stats_calculation(self):
        """Test dataset statistics calculation."""

        # Create test DataFrame
        test_data = [
            {
                "role": "user",
                "content": "Hello",
                "content_length": 5,
                "content_word_count": 1,
                "has_empty_content": False,
            },
            {
                "role": "assistant",
                "content": "Hi there!",
                "content_length": 9,
                "content_word_count": 2,
                "has_empty_content": False,
            },
            {
                "role": "user",
                "content": "Hello",
                "content_length": 5,
                "content_word_count": 1,
                "has_empty_content": False,
            },  # duplicate
        ]
        df = pd.DataFrame(test_data)

        config = DatasetConfig()
        checker = DatasetQualityChecker(config)

        try:
            stats = checker._calculate_dataset_stats(df)

            assert stats.total_entries == 3
            assert stats.unique_entries == 2  # One duplicate
            assert stats.duplicate_count == 1
            assert stats.role_distribution["user"] == 2
            assert stats.role_distribution["assistant"] == 1
            assert stats.avg_content_length == (5 + 9 + 5) / 3
            assert stats.empty_content_count == 0

        finally:
            checker.cleanup()

    def test_config_validation(self):
        """Test configuration validation."""

        # Valid config
        config = DatasetConfig(
            min_content_length=5,
            max_content_length=1000,
            allowed_roles=["user", "assistant"],
            duplicate_threshold=0.1,
        )

        assert config.min_content_length == 5
        assert config.max_content_length == 1000
        assert "user" in config.allowed_roles
        assert "assistant" in config.allowed_roles
        assert config.duplicate_threshold == 0.1

        # Test role normalization
        config = DatasetConfig(allowed_roles=[" User ", "ASSISTANT "])
        assert "user" in config.allowed_roles
        assert "assistant" in config.allowed_roles


def test_sample_data_processing():
    """Test processing of the example.json format."""

    # Sample conversation data similar to example.json
    conversations = [
        ConversationEntry(role="user", content="Please provide a detailed clinical case study..."),
        ConversationEntry(
            role="assistant",
            content="## Case Study: Acute Decompensated Heart Failure Management...",
        ),
    ]

    # Verify entries are valid
    for conv in conversations:
        assert len(conv.content) > 0
        assert conv.role in ["user", "assistant"]
        assert isinstance(conv.content, str)
        assert isinstance(conv.role, str)
