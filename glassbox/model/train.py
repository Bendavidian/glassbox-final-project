"""L3: the training loop - seeds, early stopping, checkpoints and scaler statistics.

Normalisation statistics are fitted on the training slice only and stored inside the
forecaster. Two runs with the same seed produce identical weights.

Implemented in GB-15.
"""
