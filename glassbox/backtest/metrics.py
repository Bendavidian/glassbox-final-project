"""X: forecast and trading metrics - MAE, RMSE, direction accuracy of the H-day trend,
total return, Sharpe, max drawdown, hit rate, all net of costs.

Written from scratch; reference/utils/metrics.py is unusable (see REFERENCE_AUDIT.md).
No result is reported as an absolute number: every metric carries a persistence delta,
and MSE on prices is banned as a headline metric.

Implemented in GB-19.
"""
