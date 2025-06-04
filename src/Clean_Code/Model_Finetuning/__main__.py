"""
Main entry point for the Model_Finetuning module when run as a package.
"""

import sys
from .finetuner import main

if __name__ == "__main__":
    sys.exit(main())
