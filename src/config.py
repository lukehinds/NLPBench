import json
from pathlib import Path
from typing import Optional

import yaml
from rich.console import Console

from .models import DatasetConfig

console = Console()


class ConfigManager:
    """Manages configuration loading and validation for the quality checker."""

    def __init__(self):
        self.config_paths = [
            Path.cwd() / ".nlpbench.json",
            Path.cwd() / ".nlpbench.yaml",
            Path.cwd() / ".nlpbench.yml",
            Path.home() / ".nlpbench" / "config.json",
            Path.home() / ".nlpbench" / "config.yaml",
            Path.home() / ".nlpbench" / "config.yml",
        ]

    def load_config(self, config_path: Optional[str] = None) -> DatasetConfig:
        """Load configuration from file or use defaults."""

        if config_path:
            # Use provided config path
            custom_path = Path(config_path)
            if custom_path.exists():
                return self._load_config_file(custom_path)
            else:
                console.print(
                    f"[yellow]Warning: Config file {config_path} not found. Using defaults.[/yellow]"
                )
                return DatasetConfig()

        # Try default config paths
        for default_config_path in self.config_paths:
            if default_config_path.exists():
                console.print(f"[blue]Loading config from: {default_config_path}[/blue]")
                return self._load_config_file(default_config_path)

        # No config found, use defaults
        console.print("[blue]No config file found. Using default configuration.[/blue]")
        return DatasetConfig()

    def _load_config_file(self, config_path: Path) -> DatasetConfig:
        """Load configuration from a specific file."""

        try:
            config_text = config_path.read_text(encoding="utf-8")

            if config_path.suffix.lower() in [".yaml", ".yml"]:
                config_data = yaml.safe_load(config_text)
            elif config_path.suffix.lower() == ".json":
                config_data = json.loads(config_text)
            else:
                # Try to parse as JSON first, then YAML
                try:
                    config_data = json.loads(config_text)
                except json.JSONDecodeError:
                    config_data = yaml.safe_load(config_text)

            # Validate and create config
            return DatasetConfig(**config_data)

        except Exception as e:
            console.print(f"[red]Error loading config from {config_path}: {str(e)}[/red]")
            console.print("[yellow]Using default configuration.[/yellow]")
            return DatasetConfig()

    def save_config(self, config: DatasetConfig, config_path: str, format: str = "json") -> None:
        """Save configuration to a file."""

        try:
            config_path_obj = Path(config_path)
            config_path_obj.parent.mkdir(parents=True, exist_ok=True)

            config_dict = config.model_dump()

            if format.lower() == "json":
                config_content = json.dumps(config_dict, indent=2, ensure_ascii=False)
            elif format.lower() in ["yaml", "yml"]:
                config_content = yaml.dump(
                    config_dict, default_flow_style=False, allow_unicode=True
                )
            else:
                raise ValueError(f"Unsupported format: {format}")

            config_path_obj.write_text(config_content, encoding="utf-8")
            console.print(f"[green]Config saved to: {config_path}[/green]")

        except Exception as e:
            console.print(f"[red]Error saving config: {str(e)}[/red]")
            raise

    def create_default_config(self, config_path: str, format: str = "json") -> None:
        """Create a default configuration file."""

        default_config = DatasetConfig()
        self.save_config(default_config, config_path, format)

    def get_config_template(self, format: str = "json") -> str:
        """Get a configuration template as a string."""

        default_config = DatasetConfig()
        config_dict = default_config.model_dump()

        if format.lower() == "json":
            return json.dumps(config_dict, indent=2, ensure_ascii=False)
        elif format.lower() in ["yaml", "yml"]:
            return yaml.dump(config_dict, default_flow_style=False, allow_unicode=True)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def validate_config_file(self, config_path: str) -> bool:
        """Validate a configuration file."""

        try:
            _config = self._load_config_file(Path(config_path))
            console.print(f"[green]Config file {config_path} is valid[/green]")
            return True
        except Exception as e:
            console.print(f"[red]Config file {config_path} is invalid: {str(e)}[/red]")
            return False


# Global config manager instance
config_manager = ConfigManager()


def load_config(config_path: Optional[str] = None) -> DatasetConfig:
    """Convenience function to load configuration."""
    return config_manager.load_config(config_path)


def save_default_config(config_path: str = ".nlpbench.json") -> None:
    """Convenience function to create a default config file."""
    config_manager.create_default_config(config_path)


def get_config_example() -> str:
    """Get an example configuration as JSON string."""
    return config_manager.get_config_template("json")
