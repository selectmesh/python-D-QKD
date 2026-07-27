#!/usr/bin/env python3
"""
Reproducible calibration-sensitivity test for sector-conditioned
three-intensity decoy-state QKD.

The script:
  1. loads a measured/simulated sector profile at one distance;
  2. constructs statistical gain intervals;
  3. widens each interval by eta_m * mu * delta_rel;
  4. solves a linear program in every sector to lower-bound Y_1;
  5. reports the reduction of the certified signal single-photon contribution.

Scientific scope
----------------
This script certifies only the numerical statement produced by the declared
decoy-state LP and input profile. It is not a complete QKD security proof.

Required CSV columns:
  sector, weight, eta, Q_s, Q_d, Q_v

Here eta is the total sector transmittance used in the calibration radius.
Q_s, Q_d, Q_v are observed or modelled gains for mu_s, mu_d, mu_v.

Optional CSV columns:
  N_s, N_d, N_v
If absent, trial counts are generated from --n, --gamma and --p-intensity.

Example:
  python calibration_decoy_reproduction.py \
      --profile sector_profile_50km.csv \
      --delta-rel 1e-4 1e-3
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.optimize import linprog
from scipy.special import gammainc


@dataclass(frozen=True)
class Sector:
    index: int
    weight: float
    eta: float
    gains: np.ndarray
    trials: np.ndarray


def poisson_weights(mu: float, nmax: int) -> np.ndarray:
    n = np.arange(nmax + 1)
    # Stable recurrence is unnecessary at the small intensities used here.
    fact = np.array([math.factorial(int(k)) for k in n], dtype=float)
    return np.exp(-mu) * np.power(mu, n) / fact


def poisson_tail(mu: float, nmax: int) -> float:
    # P[N > nmax] for N~Poisson(mu).
    return float(gammainc(nmax + 1, mu))


def hoeffding_radius(num_trials: float, eps: float) -> float:
    if num_trials <= 0:
        raise ValueError("Every gain must have a positive number of trials.")
    if not (0.0 < eps < 1.0):
        raise ValueError("eps must lie in (0,1).")
    return math.sqrt(math.log(2.0 / eps) / (2.0 * num_trials))


def y1_lower_bound(
    mus: np.ndarray,
    q_lower: np.ndarray,
    q_upper: np.ndarray,
    nmax: int,
) -> float:
    """
    Minimise Y_1 subject to
      Q_mu = sum_{n=0}^{nmax} P_mu(n) Y_n + r_mu,
      0 <= Y_n <= 1, 0 <= r_mu <= PoissonTail(mu).
    Eliminating r_mu yields:
      sum P_mu(n)Y_n <= Q_upper,
      sum P_mu(n)Y_n >= Q_lower - tail.
    """
    dim = nmax + 1
    c = np.zeros(dim)
    c[1] = 1.0

    aub = []
    bub = []
    for mu, ql, qu in zip(mus, q_lower, q_upper):
        p = poisson_weights(float(mu), nmax)
        tail = poisson_tail(float(mu), nmax)

        aub.append(p)
        bub.append(qu)

        aub.append(-p)
        bub.append(-(ql - tail))

    result = linprog(
        c,
        A_ub=np.asarray(aub),
        b_ub=np.asarray(bub),
        bounds=[(0.0, 1.0)] * dim,
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"Decoy LP infeasible: {result.message}")
    return float(result.x[1])


def build_intervals(
    sector: Sector,
    mus: np.ndarray,
    eps_each: float,
    delta_rel: float,
) -> tuple[np.ndarray, np.ndarray]:
    stat = np.array(
        [hoeffding_radius(N, eps_each) for N in sector.trials], dtype=float
    )
    calibration = sector.eta * mus * delta_rel
    radius = stat + calibration
    lower = np.clip(sector.gains - radius, 0.0, 1.0)
    upper = np.clip(sector.gains + radius, 0.0, 1.0)
    return lower, upper


def certified_signal_single_photon(
    sectors: Iterable[Sector],
    mus: np.ndarray,
    eps_each: float,
    delta_rel: float,
    nmax: int,
) -> tuple[float, list[float]]:
    p1_signal = math.exp(-mus[0]) * mus[0]
    sector_y1 = []
    total = 0.0
    for s in sectors:
        ql, qu = build_intervals(s, mus, eps_each, delta_rel)
        y1 = y1_lower_bound(mus, ql, qu, nmax)
        sector_y1.append(y1)
        total += s.weight * p1_signal * y1
    return total, sector_y1


def load_profile(
    path: Path,
    n_total: float,
    gamma: float,
    p_intensity: np.ndarray,
) -> list[Sector]:
    sectors: list[Sector] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"sector", "weight", "eta", "Q_s", "Q_d", "Q_v"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing CSV columns: {sorted(missing)}")

        for row in reader:
            weight = float(row["weight"])
            gains = np.array(
                [float(row["Q_s"]), float(row["Q_d"]), float(row["Q_v"])],
                dtype=float,
            )
            if all(k in row and row[k] not in ("", None) for k in ("N_s", "N_d", "N_v")):
                trials = np.array(
                    [float(row["N_s"]), float(row["N_d"]), float(row["N_v"])],
                    dtype=float,
                )
            else:
                trials = n_total * gamma * weight * p_intensity

            sectors.append(
                Sector(
                    index=int(row["sector"]),
                    weight=weight,
                    eta=float(row["eta"]),
                    gains=gains,
                    trials=trials,
                )
            )

    total_weight = sum(s.weight for s in sectors)
    if not math.isclose(total_weight, 1.0, rel_tol=0.0, abs_tol=1e-8):
        raise ValueError(f"Sector weights must sum to 1; obtained {total_weight}.")
    return sectors


def profile_from_paper_model(
    distance_km: float,
    mus: np.ndarray,
    n_total: float,
    gamma: float,
    p_intensity: np.ndarray,
) -> tuple[list[Sector], float]:
    """
    Build the sector profile from the *declared channel model of the
    manuscript*, i.e. the same module that produces the production curves
    (min_entropy_distance.sector_obs).  This makes the calibration-sensitivity
    number a reproduction of the paper's own operating point rather than a
    synthetic illustration.

    The declared model uses a sector-independent transmittance
    eta = eta_det * 10^(-alpha L/10) and dark-count floor Y0 = 2 p_d, with
    Q_mu = Y0 + 1 - exp(-eta mu); sector heterogeneity enters the *error*
    profile only, which does not appear in the Y_1 linear program.
    """
    import min_entropy_distance as med

    eta = med.ETA_D * 10.0 ** (-med.ALPHA_DB * distance_km / 10.0)
    y0 = 2.0 * med.P_D
    gains = np.array([y0 + 1.0 - math.exp(-eta * mu) for mu in mus], dtype=float)

    weight = 1.0 / med.M
    sectors = [
        Sector(
            index=m,
            weight=weight,
            eta=eta,
            gains=gains.copy(),
            trials=n_total * gamma * weight * p_intensity,
        )
        for m in range(med.M)
    ]
    return sectors, float(med.EPS_PE)


def write_example_profile(path: Path, distance_km: float, alpha_db_km: float,
                          eta_det: float, pdark: float, sectors_count: int,
                          heterogeneity: float, mus: np.ndarray) -> None:
    """
    Generates an explicitly synthetic profile for code testing only.
    It must not be presented as experimental or paper-reproduction data.
    """
    eta_mean = eta_det * 10.0 ** (-alpha_db_km * distance_km / 10.0)
    rows = []
    for m in range(sectors_count):
        theta = 2.0 * math.pi * m / sectors_count
        eta_m = eta_mean * (1.0 + heterogeneity * math.cos(theta))
        gains = [1.0 - (1.0 - pdark) ** 2 * math.exp(-eta_m * mu) for mu in mus]
        rows.append([m, 1.0 / sectors_count, eta_m, *gains])

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sector", "weight", "eta", "Q_s", "Q_d", "Q_v"])
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--profile", type=Path, help="CSV sector profile at 50 km.")
    p.add_argument("--paper-model", action="store_true",
                   help="Build the profile from the manuscript's declared "
                        "channel model (min_entropy_distance.py) instead of a CSV.")
    p.add_argument("--make-example-profile", type=Path,
                   help="Write a synthetic example CSV and exit.")
    p.add_argument("--n", type=float, default=1e12)
    p.add_argument("--gamma", type=float, default=0.05)
    p.add_argument("--mu", nargs=3, type=float, default=(0.6, 0.1, 0.001),
                   metavar=("MU_S", "MU_D", "MU_V"))
    p.add_argument("--p-intensity", nargs=3, type=float,
                   default=(0.80, 0.15, 0.05),
                   metavar=("P_S", "P_D", "P_V"))
    p.add_argument("--delta-rel", nargs="+", type=float,
                   default=(1e-4, 1e-3))
    p.add_argument("--eps-pe", type=float, default=1e-10,
                   help="Total failure budget allocated to gain intervals.")
    p.add_argument("--nmax", type=int, default=12)
    p.add_argument("--distance-km", type=float, default=50.0)
    p.add_argument("--alpha-db-km", type=float, default=0.2)
    p.add_argument("--eta-det", type=float, default=0.85)
    p.add_argument("--pdark", type=float, default=1e-8)
    p.add_argument("--sectors", type=int, default=8)
    p.add_argument("--heterogeneity", type=float, default=0.10)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    mus = np.asarray(args.mu, dtype=float)
    p_intensity = np.asarray(args.p_intensity, dtype=float)
    if not math.isclose(float(p_intensity.sum()), 1.0, abs_tol=1e-12):
        raise ValueError("--p-intensity values must sum to 1.")

    if args.make_example_profile:
        write_example_profile(
            args.make_example_profile,
            args.distance_km,
            args.alpha_db_km,
            args.eta_det,
            args.pdark,
            args.sectors,
            args.heterogeneity,
            mus,
        )
        print(f"Wrote synthetic test profile: {args.make_example_profile}")
        print("This profile is illustrative and is not paper-reproduction data.")
        return 0

    if args.paper_model:
        sectors, eps_total = profile_from_paper_model(
            args.distance_km, mus, args.n, args.gamma, p_intensity
        )
        source = (f"declared channel model of the manuscript "
                  f"(min_entropy_distance.py) at L={args.distance_km:g} km")
    elif args.profile is not None:
        sectors = load_profile(args.profile, args.n, args.gamma, p_intensity)
        eps_total = args.eps_pe
        source = str(args.profile)
    else:
        print(
            "ERROR: give --paper-model (reproduces the manuscript's operating\n"
            "point from its own declared channel model) or --profile CSV.\n"
            "Use --make-example-profile only to test the implementation.",
            file=sys.stderr,
        )
        return 2

    # Bonferroni allocation across all sector/intensity gain intervals.
    eps_each = eps_total / (len(sectors) * len(mus))

    baseline, baseline_y1 = certified_signal_single_photon(
        sectors, mus, eps_each, 0.0, args.nmax
    )
    print(f"Profile: {source}")
    print(f"sectors={len(sectors)}, n={args.n:.6g}, gamma={args.gamma:g}")
    print(f"mu={tuple(mus)}, eps_each={eps_each:.3e}, nmax={args.nmax}")
    print(f"baseline certified signal single-photon contribution = {baseline:.12e}")
    print("baseline Y1 lower bounds:")
    for s, y1 in zip(sectors, baseline_y1):
        print(f"  sector {s.index:2d}: {y1:.12e}")

    if baseline <= 0:
        raise RuntimeError("Baseline certified contribution is non-positive.")

    print("\nCalibration sensitivity:")
    for delta in args.delta_rel:
        value, _ = certified_signal_single_photon(
            sectors, mus, eps_each, delta, args.nmax
        )
        reduction = 100.0 * (baseline - value) / baseline
        print(
            f"  delta_rel={delta:.3e}: contribution={value:.12e}, "
            f"reduction={reduction:.6f}%"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
