"""L2 THE KEYSTONE: the only place in the codebase where a model input window is
assembled (spec 3.3).

Offline training and the live loop call this same function with the same config, which
is what structurally guarantees the model sees the same thing in both worlds. Do not
create a second assembly path. Produces a WindowBatch.

Implemented in GB-9. Train/live parity is asserted in GB-27.
"""
