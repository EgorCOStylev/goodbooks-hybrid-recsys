"""Run the full pipeline from the repository root: python run_pipeline.py --help"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))

from recsys.run import main

if __name__ == '__main__':
    main()
