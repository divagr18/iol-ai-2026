import json
import sys
from pathlib import Path

def patch_tokenizer_config(model_dir: str):
    config_path = Path(model_dir) / "tokenizer_config.json"
    if not config_path.exists():
        print(f"No tokenizer_config.json at {config_path}")
        return
    
    with open(config_path, "r") as f:
        config = json.load(f)
    
    # Fix extra_special_tokens if it's a list instead of dict
    extra = config.get("extra_special_tokens")
    if isinstance(extra, list):
        print(f"Patching extra_special_tokens from list to dict ({len(extra)} items)")
        config["extra_special_tokens"] = {str(i): tok for i, tok in enumerate(extra)}
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        print("Patched successfully")
    else:
        print("extra_special_tokens is already correct")

if __name__ == "__main__":
    patch_tokenizer_config(sys.argv[1])
