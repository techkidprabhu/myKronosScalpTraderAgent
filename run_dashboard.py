#!/usr/bin/env python3
"""
Entry point to launch the Kronos Scalp Trading Agent web dashboard.
Usage:
    python run_dashboard.py --port 5010
"""
from dashboard.app import main

if __name__ == "__main__":
    main()
