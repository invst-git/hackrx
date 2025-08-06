# config.py

# --- Chunking Parameters ---
# The maximum number of tokens allowed in a single text chunk.
# This is set to work well with models like text-embedding-3-large.
MAX_CHUNK_TOKENS = 1000

# The number of tokens that will overlap between consecutive chunks.
# This helps maintain context and prevents losing information at chunk boundaries.
CHUNK_OVERLAP_TOKENS = 20
