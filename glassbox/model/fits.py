"""L3: `FITSForecaster` - RIN, rFFT, low-pass filter, complex linear layer, irFFT
(spec 6.2).

One complex weight matrix learns an amplitude gain and a phase shift per retained
frequency. The irFFT output must be scaled by (L + H) / L; see spec 6.3.

Implemented in GB-41. Amplitude scaling and its sinusoid test are GB-42.
"""
