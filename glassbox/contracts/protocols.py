"""L0: the `Forecaster` protocol of spec 4.3 - fit, predict, explain, save, load.

Every model implements exactly this, so nothing downstream knows which model it holds.
A model is not integrated until it passes tests/test_forecaster_contract.py unchanged.

Implemented in GB-3.
"""
