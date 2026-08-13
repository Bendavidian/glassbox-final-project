"""L5/L7 entry point: the autonomous decision cycle (spec 3.5).

Polls every 60 seconds during US market hours, then walks data, builder, forecast,
decision, explanation, execution and the decision record.

Implemented in GB-26. Fault handling and restart safety are GB-39.
"""
