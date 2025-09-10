import json
from typing import Any, Optional

import pandas as pd
from datasets import load_dataset
from rich.console import Console
from tqdm import tqdm

from .models import ConversationEntry, HuggingFaceDatasetInfo

console = Console()


class DatasetLoader:
    """Loads and processes Hugging Face datasets for quality assessment."""

    def __init__(self, dataset_info: HuggingFaceDatasetInfo):
        self.dataset_info = dataset_info
        self.raw_dataset = None
        self.processed_data: list[ConversationEntry] = []

    def load_dataset(self) -> None:
        """Load dataset from Hugging Face Hub."""
        try:
            console.print(f"[blue]Loading dataset: {self.dataset_info.repo_id}[/blue]")

            self.raw_dataset = load_dataset(
                self.dataset_info.repo_id,
                name=self.dataset_info.config_name,
                split=self.dataset_info.split,
                revision=self.dataset_info.revision,
                cache_dir=self.dataset_info.cache_dir,
                token=self.dataset_info.token,
            )

            console.print(f"[green]Successfully loaded {len(self.raw_dataset)} examples[/green]")

        except Exception as e:
            console.print(f"[red]Error loading dataset: {str(e)}[/red]")
            raise

    def _extract_conversation_from_row(self, row: dict[str, Any]) -> list[ConversationEntry]:
        """Extract conversation entries from a dataset row."""
        conversations = []

        # Handle different dataset formats
        if "conversations" in row:
            # Standard conversations format
            for conv in row["conversations"]:
                if isinstance(conv, dict) and "role" in conv and "content" in conv:
                    try:
                        conversations.append(
                            ConversationEntry(role=conv["role"], content=str(conv["content"]))
                        )
                    except Exception as e:
                        console.print(f"[yellow]Warning: Invalid conversation entry: {e}[/yellow]")

        elif "messages" in row:
            # Messages format (common in chat datasets)
            for msg in row["messages"]:
                if isinstance(msg, dict) and "role" in msg and "content" in msg:
                    try:
                        conversations.append(
                            ConversationEntry(role=msg["role"], content=str(msg["content"]))
                        )
                    except Exception as e:
                        console.print(f"[yellow]Warning: Invalid message entry: {e}[/yellow]")

        elif "instruction" in row and "output" in row:
            # Instruction-following format
            try:
                conversations.append(
                    ConversationEntry(role="user", content=str(row["instruction"]))
                )
                conversations.append(
                    ConversationEntry(role="assistant", content=str(row["output"]))
                )
                if "input" in row and row["input"]:
                    # Insert input as context
                    conversations.insert(
                        1, ConversationEntry(role="system", content=str(row["input"]))
                    )
            except Exception as e:
                console.print(f"[yellow]Warning: Invalid instruction entry: {e}[/yellow]")

        elif "question" in row and "answer" in row:
            # Q&A format
            try:
                conversations.append(ConversationEntry(role="user", content=str(row["question"])))
                conversations.append(
                    ConversationEntry(role="assistant", content=str(row["answer"]))
                )
            except Exception as e:
                console.print(f"[yellow]Warning: Invalid Q&A entry: {e}[/yellow]")

        elif "prompt" in row and "response" in row:
            # Prompt-response format
            try:
                conversations.append(ConversationEntry(role="user", content=str(row["prompt"])))
                conversations.append(
                    ConversationEntry(role="assistant", content=str(row["response"]))
                )
            except Exception as e:
                console.print(f"[yellow]Warning: Invalid prompt-response entry: {e}[/yellow]")

        elif "text" in row:
            # Single text format - treat as assistant response
            try:
                conversations.append(ConversationEntry(role="assistant", content=str(row["text"])))
            except Exception as e:
                console.print(f"[yellow]Warning: Invalid text entry: {e}[/yellow]")

        else:
            # Try to handle as JSON string if it looks like conversation data
            for key, value in row.items():
                if isinstance(value, str):
                    try:
                        parsed = json.loads(value)
                        if isinstance(parsed, list):
                            for item in parsed:
                                if isinstance(item, dict) and "role" in item and "content" in item:
                                    conversations.append(
                                        ConversationEntry(
                                            role=item["role"], content=str(item["content"])
                                        )
                                    )
                        break
                    except json.JSONDecodeError:
                        continue

            # Fallback: treat any text-like fields as generic content
            if not conversations:
                conversations.extend(self._extract_generic_text_content(row))

        return conversations

    def _extract_generic_text_content(self, row: dict[str, Any]) -> list[ConversationEntry]:
        """Extract text content from any dataset format as a fallback."""
        conversations = []

        # Common text fields to look for
        text_fields = [
            "text",
            "content",
            "body",
            "description",
            "summary",
            "title",
            "review",
            "comment",
            "message",
            "sentence",
            "paragraph",
            "resume",
            "job_description",
            "abstract",
            "article",
            "document",
        ]

        # Look for fields that likely contain substantial text content
        for key, value in row.items():
            if value is None:
                continue

            # Convert to string and check if it's substantial text
            text_content = str(value).strip()

            # Skip very short content, numbers, or URLs
            if len(text_content) < 10:
                continue
            if text_content.isdigit():
                continue
            if text_content.startswith(("http://", "https://", "ftp://")):
                continue

            # Determine role based on field name
            role = (
                "user"
                if any(
                    term in key.lower()
                    for term in ["input", "question", "query", "prompt", "request"]
                )
                else "document"
            )

            try:
                conversations.append(ConversationEntry(role=role, content=text_content))
            except Exception as e:
                console.print(f"[yellow]Warning: Could not process field '{key}': {e}[/yellow]")

        # If we still have no content, try to concatenate all string fields
        if not conversations:
            all_text = []
            for key, value in row.items():
                if isinstance(value, str) and len(str(value).strip()) > 0:
                    all_text.append(f"{key}: {value}")

            if all_text:
                combined_text = " | ".join(all_text)
                try:
                    conversations.append(ConversationEntry(role="document", content=combined_text))
                except Exception as e:
                    console.print(f"[yellow]Warning: Could not process combined text: {e}[/yellow]")

        return conversations

    def process_dataset(self) -> list[ConversationEntry]:
        """Process the raw dataset into conversation entries."""
        if self.raw_dataset is None:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")

        console.print("[blue]Processing dataset entries...[/blue]")

        self.processed_data = []

        with tqdm(total=len(self.raw_dataset), desc="Processing entries") as pbar:
            for row in self.raw_dataset:
                conversations = self._extract_conversation_from_row(row)
                self.processed_data.extend(conversations)
                pbar.update(1)

        console.print(f"[green]Processed {len(self.processed_data)} conversation entries[/green]")
        return self.processed_data

    def to_dataframe(self) -> pd.DataFrame:
        """Convert processed data to pandas DataFrame for Great Expectations."""
        if not self.processed_data:
            raise ValueError("No processed data available. Call process_dataset() first.")

        data_dicts = []
        for entry in self.processed_data:
            data_dicts.append(
                {
                    "role": entry.role,
                    "content": entry.content,
                    "content_length": len(entry.content),
                    "content_word_count": len(entry.content.split()),
                    "has_empty_content": len(entry.content.strip()) == 0,
                }
            )

        return pd.DataFrame(data_dicts)

    def get_sample_data(self, n: int = 5) -> list[dict[str, Any]]:
        """Get sample data for inspection."""
        if not self.processed_data:
            return []

        sample_size = min(n, len(self.processed_data))
        sample_entries = self.processed_data[:sample_size]

        return [
            {
                "role": entry.role,
                "content": entry.content[:200] + "..."
                if len(entry.content) > 200
                else entry.content,
                "content_length": len(entry.content),
            }
            for entry in sample_entries
        ]

    def get_dataset_info(self) -> dict[str, Any]:
        """Get basic information about the loaded dataset."""
        if self.raw_dataset is None:
            return {}

        info = {
            "repo_id": self.dataset_info.repo_id,
            "split": self.dataset_info.split,
            "raw_entries": len(self.raw_dataset),
            "processed_entries": len(self.processed_data),
            "features": list(self.raw_dataset.features.keys())
            if hasattr(self.raw_dataset, "features")
            else [],
        }

        if self.processed_data:
            role_counts = {}
            for entry in self.processed_data:
                role_counts[entry.role] = role_counts.get(entry.role, 0) + 1
            info["role_distribution"] = role_counts

        return info


def load_and_process_dataset(
    repo_id: str,
    config_name: Optional[str] = None,
    split: str = "train",
    revision: Optional[str] = None,
    cache_dir: Optional[str] = None,
    token: Optional[str] = None,
) -> DatasetLoader:
    """Convenience function to load and process a dataset."""

    dataset_info = HuggingFaceDatasetInfo(
        repo_id=repo_id,
        config_name=config_name,
        split=split,
        revision=revision,
        cache_dir=cache_dir,
        token=token,
    )

    loader = DatasetLoader(dataset_info)
    loader.load_dataset()
    loader.process_dataset()

    return loader
