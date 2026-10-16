"""Disk cache shared by all tool wrappers.

Every API response is cached so repeated runs are fast, stay under rate
limits, and give the same results for the final notebook.
"""

from joblib import Memory

from src.config import CACHE_DIR

memory = Memory(CACHE_DIR, verbose=0)
