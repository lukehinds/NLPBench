from pathlib import Path
from typing import Any, Optional

import kagglehub
import pandas as pd
from rich.console import Console

from .models import ConversationEntry, KaggleDatasetInfo

console = Console()


class KaggleDatasetLoader:
    """Loads and processes Kaggle datasets for quality assessment."""

    def __init__(self, dataset_info: KaggleDatasetInfo):
        self.dataset_info = dataset_info
        self.downloaded_path: Optional[Path] = None
        self.processed_data: list[ConversationEntry] = []
        self.dataframes: list[pd.DataFrame] = []

    def download_dataset(self) -> Path:
        """Download dataset from Kaggle."""
        try:
            console.print(
                f"[blue]Downloading Kaggle dataset: {self.dataset_info.dataset_id}[/blue]"
            )

            # Download using kagglehub
            # Note: kagglehub doesn't support version parameter directly in dataset_download
            # Version needs to be included in the dataset handle like "owner/dataset/versions/N"
            dataset_handle = self.dataset_info.dataset_id
            if self.dataset_info.version:
                # Append version to handle if not already included
                if "/versions/" not in dataset_handle:
                    dataset_handle = f"{dataset_handle}/versions/{self.dataset_info.version}"

            path = kagglehub.dataset_download(
                dataset_handle,
                path=self.dataset_info.path,
                force_download=self.dataset_info.force_download,
            )

            self.downloaded_path = Path(path)
            console.print(f"[green]Dataset downloaded to: {self.downloaded_path}[/green]")

            return self.downloaded_path

        except Exception as e:
            console.print(f"[red]Error downloading Kaggle dataset: {str(e)}[/red]")
            raise

    def load_data_files(self) -> list[pd.DataFrame]:
        """Load data files from the downloaded dataset."""
        if self.downloaded_path is None:
            raise ValueError("Dataset not downloaded. Call download_dataset() first.")

        console.print("[blue]Loading data files...[/blue]")

        dataframes = []

        # If specific path is provided, load only that file
        if self.dataset_info.path:
            target_path = self.downloaded_path / self.dataset_info.path
            if target_path.exists():
                df = self._load_file(target_path)
                if df is not None:
                    dataframes.append(df)
            else:
                console.print(f"[yellow]Warning: Specified path not found: {target_path}[/yellow]")
        else:
            # Load all supported files
            for file_path in self.downloaded_path.rglob("*"):
                if file_path.is_file() and self._is_supported_file(file_path):
                    df = self._load_file(file_path)
                    if df is not None:
                        dataframes.append(df)

        self.dataframes = dataframes
        total_rows = sum(len(df) for df in dataframes)
        console.print(
            f"[green]Loaded {len(dataframes)} file(s) with {total_rows:,} total rows[/green]"
        )

        return dataframes

    def _is_supported_file(self, file_path: Path) -> bool:
        """Check if the file type is supported."""
        supported_extensions = {".csv", ".json", ".jsonl", ".parquet", ".tsv", ".txt"}
        return file_path.suffix.lower() in supported_extensions

    def _load_file(self, file_path: Path) -> Optional[pd.DataFrame]:
        """Load a single file into a DataFrame."""
        try:
            extension = file_path.suffix.lower()

            if extension == ".csv":
                return pd.read_csv(file_path)
            elif extension == ".json":
                return pd.read_json(file_path)
            elif extension == ".jsonl":
                return pd.read_json(file_path, lines=True)
            elif extension == ".parquet":
                return pd.read_parquet(file_path)
            elif extension == ".tsv":
                return pd.read_csv(file_path, sep="\t")
            elif extension == ".txt":
                # Read as single column text
                with open(file_path, encoding="utf-8") as f:
                    lines = [line.strip() for line in f.readlines() if line.strip()]
                return pd.DataFrame({"text": lines})
            else:
                console.print(f"[yellow]Unsupported file type: {file_path}[/yellow]")
                return None

        except Exception as e:
            console.print(f"[yellow]Error loading {file_path}: {str(e)}[/yellow]")
            return None

    def process_dataframes(self) -> list[ConversationEntry]:
        """Process loaded DataFrames into conversation entries."""
        if not self.dataframes:
            raise ValueError("No data loaded. Call load_data_files() first.")

        console.print("[blue]Processing dataset entries...[/blue]")

        self.processed_data = []

        for i, df in enumerate(self.dataframes):
            console.print(
                f"[dim]Processing DataFrame {i + 1}/{len(self.dataframes)} ({len(df)} rows)[/dim]"
            )

            for _, row in df.iterrows():
                conversations = self._extract_conversation_from_row(row.to_dict())
                self.processed_data.extend(conversations)

        console.print(f"[green]Processed {len(self.processed_data)} conversation entries[/green]")
        return self.processed_data

    def _extract_conversation_from_row(self, row: dict[str, Any]) -> list[ConversationEntry]:
        """Extract conversation entries from a dataset row."""
        conversations = []

        # Handle different dataset formats (same logic as HF loader)
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
            # Single text format - treat as document
            try:
                conversations.append(ConversationEntry(role="document", content=str(row["text"])))
            except Exception as e:
                console.print(f"[yellow]Warning: Invalid text entry: {e}[/yellow]")

        else:
            # Fallback: treat any text-like fields as generic content
            conversations.extend(self._extract_generic_text_content(row))

        return conversations

    def _extract_generic_text_content(self, row: dict[str, Any]) -> list[ConversationEntry]:
        """Extract text content from any dataset format as a fallback."""
        conversations = []

        # Look for fields that likely contain substantial text content
        for key, value in row.items():
            if value is None or pd.isna(value):
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
                if not pd.isna(value) and isinstance(value, str) and len(str(value).strip()) > 0:
                    all_text.append(f"{key}: {value}")

            if all_text:
                combined_text = " | ".join(all_text)
                try:
                    conversations.append(ConversationEntry(role="document", content=combined_text))
                except Exception as e:
                    console.print(f"[yellow]Warning: Could not process combined text: {e}[/yellow]")

        return conversations

    def to_dataframe(self) -> pd.DataFrame:
        """Convert processed data to pandas DataFrame for quality checking."""
        if not self.processed_data:
            raise ValueError("No processed data available. Call process_dataframes() first.")

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
        info = {
            "dataset_id": self.dataset_info.dataset_id,
            "version": self.dataset_info.version or "latest",
            "downloaded_path": str(self.downloaded_path) if self.downloaded_path else None,
            "files_loaded": len(self.dataframes),
            "processed_entries": len(self.processed_data),
        }

        if self.processed_data:
            role_counts = {}
            for entry in self.processed_data:
                role_counts[entry.role] = role_counts.get(entry.role, 0) + 1
            info["role_distribution"] = role_counts

        return info
