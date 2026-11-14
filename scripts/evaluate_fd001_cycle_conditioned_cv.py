"""Run the grouped cycle-conditioned calibration evaluation on FD001."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_fd004_cycle_conditioned_cv import main


if __name__ == "__main__":
    main("FD001")
