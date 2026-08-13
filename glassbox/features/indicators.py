"""L2: the pure trailing-window indicators rsi14, vol_z, mom10 and ma_dist20.

Every value at time t uses only bars <= t. Centred windows are forbidden here and are
tested for in GB-10. Each function is pure: DataFrame in, DataFrame out.

Implemented in GB-8.
"""
