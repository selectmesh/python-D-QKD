# Supplementary code: Sector-Conditioned Finite-Key Analysis
Requirements: Python 3.10+, numpy, scipy, matplotlib, cvxpy (Clarabel solver).
  pip install numpy scipy matplotlib cvxpy

Reproduction commands (each regenerates its figures/CSVs):
  python3 run_all.py        # static profile B1/B3/B4, M-scan, decoy LPs
  python3 run_scenarios.py  # adaptive partition + dynamic channel
  python3 run_selection.py  # public sector selection (28 -> 168 km)
  python3 unified_decoy.py  # unified Choi-state construction self-tests
  python3 exact_talpha.py   # exact second-order verification
  python3 decoy_geat.py     # dual extraction -> certified f -> full GEAT layer

Map to the paper:
  fig_qubit_adaptive    <- run_scenarios.py   (Sec. VII A)
  fig_dynamic_selection <- run_selection.py   (Sec. VII A)
  fig_unified_decoy     <- unified_decoy sweep (Sec. VII B)
  fig_decoy_geat        <- decoy_geat.py      (Sec. VII B, dual->GEAT)
  calibration_example   <- worked example     (Sec. IX A)
