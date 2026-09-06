"""CLI script to run Phase 0 — synthetic dataset generation.

Usage:
    python scripts/generate_synthetic.py
    python scripts/generate_synthetic.py --force-download
"""

import sys
from pathlib import Path

# Ensure repo root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.xbra.synthetic.generator import SyntheticDatasetGenerator
from config.settings import RANDOM_SEED
from loguru import logger

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--force-download", action="store_true")
    args = parser.parse_args()

    logger.remove()
    logger.add(sys.stderr, level="INFO", colorize=True,
               format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}")

    gen = SyntheticDatasetGenerator(seed=RANDOM_SEED)
    profiles = gen.generate(force_download=args.force_download)
    logger.success("Phase 0 complete — {} investor profiles generated.", len(profiles))
