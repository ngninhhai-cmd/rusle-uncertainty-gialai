"""Configuration loader."""
from pathlib import Path
import yaml


class Config:
    """Load and manage YAML configuration."""

    def __init__(self, config_path=None):
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
        self.config_path = Path(config_path)
        with open(self.config_path, "r", encoding="utf-8") as f:
            self._cfg = yaml.safe_load(f)
        self._resolve_paths()
        self._create_output_dirs()

    def _resolve_paths(self):
        root = Path(self._cfg["paths"]["data_root"])
        for key in ["factors_30m"]:
            for subkey, rel in list(self._cfg["paths"][key].items()):
                self._cfg["paths"][key][subkey] = str(root / rel)
        for key in ["K_250m", "SER_250m", "LS_capped"]:
            if key in self._cfg["paths"]:
                self._cfg["paths"][key] = str(root / self._cfg["paths"][key])

    def _create_output_dirs(self):
        out = Path(self._cfg["paths"]["output_root"])
        for sub in ["figures", "tables", "logs", "rasters"]:
            (out / sub).mkdir(parents=True, exist_ok=True)

    def __getitem__(self, key):
        return self._cfg[key]

    def get(self, *keys, default=None):
        val = self._cfg
        for k in keys:
            if isinstance(val, dict) and k in val:
                val = val[k]
            else:
                return default
        return val