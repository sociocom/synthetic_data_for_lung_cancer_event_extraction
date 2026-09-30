import json
from pathlib import Path

def main() -> None:
    base_dir = Path(__file__).resolve().parent.parent / "data" / "raw" / "simu_lung_cancer"
    reactions: set[str] = set()

    for path in sorted(base_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            raise RuntimeError(f"Failed to read JSON: {path}") from e

        if not isinstance(data, list):
            continue

        for obj in data:
            if isinstance(obj, dict) and "Reaction" in obj and isinstance(obj["Reaction"], str):
                r = obj["Reaction"].strip()
                if r:
                    reactions.add(r)

    # stable output
    print(json.dumps(sorted(reactions), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
