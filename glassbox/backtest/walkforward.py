"""X: walk-forward fold generation with strict boundaries.

Train 24 months, validate 3, test 3, roll 3. No timestamp appears in one fold's train
and another's test. Cross-validation is invalid for time series and is never used.

Implemented in GB-17.
"""
