#!/usr/bin/env python3
"""Entry point for the standalone Crazyflie mocap / visual-matching workflow."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / 'scripts'))
from pipeline_cli import main

if __name__ == '__main__':
    main()
