"""L1: data quality report per symbol - gaps, NaNs, split artefacts and NYSE calendar
alignment.

Missing data is never forward-filled across a gap greater than one bar without an
explicit flag.

Implemented in GB-5.
"""
