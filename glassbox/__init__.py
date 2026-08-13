"""GlassBox Trader - an autonomous US equities trading system that explains every
decision it makes with exact, algebraic attribution.

The package is layered L0 (config and contracts) through L7 (observability), with a
cross-cutting validation harness. Data flows strictly upward: no module may import
from a layer above it. The layer contract is enforced by import-linter in GB-3.
"""
