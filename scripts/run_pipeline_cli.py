"""CLI: PYTHONPATH=. python scripts/run_pipeline_cli.py \"message\"."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline import process_message


def main() -> None:
    """Print PipelineResult JSON for a single message from argv."""
    if len(sys.argv) < 2:
        print('Usage: python scripts/run_pipeline_cli.py "your message"')
        sys.exit(1)
    message = sys.argv[1]
    result = process_message(message)
    print(json.dumps(result.as_dict(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
