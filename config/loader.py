
import os
import yaml

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")


def load_config(path: str = _CONFIG_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    cfg = load_config()
    print("Tickers:", cfg["data"]["tickers"])
    print("Date Interval:", cfg["data"]["start_date"], "->", cfg["data"]["end_date"])
    print("Validation Method:", cfg["modeling"]["validation"]["method"])
