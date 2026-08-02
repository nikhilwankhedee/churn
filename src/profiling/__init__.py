"""
Data profiling: comprehensive dataset inspection before training.

Generates a structured report covering:
- Dataset overview (rows, columns, memory)
- Customer and transaction statistics
- Timestamp analysis
- Missing value patterns
- Duplicate detection
- Feature type classification
- Constant/near-constant feature detection
- Class imbalance assessment
- High-correlation detection
- Data leakage heuristics

Can be used standalone or integrated into the pipeline.
"""
from src.profiling.profiler import profile_dataset, DatasetProfile
