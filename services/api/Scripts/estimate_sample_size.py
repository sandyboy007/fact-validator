"""
Estimate sample size requirements for comparing two accuracies.

This script uses a normal approximation for two-proportion power analysis.
It is intended for planning benchmark size before running evaluation.

Usage examples:
  python Scripts/estimate_sample_size.py --p1 0.714 --p2 0.429
  python Scripts/estimate_sample_size.py --p1 0.714 --p2 0.571 --power 0.9
"""

from __future__ import annotations

import argparse
import math


def _normal_inv_cdf(p: float) -> float:
    # Acklam approximation for inverse normal CDF.
    # Accurate enough for statistical planning use.
    if p <= 0.0 or p >= 1.0:
        raise ValueError("p must be in (0, 1)")

    a = [
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    ]
    b = [
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    ]
    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    ]
    d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    ]

    plow = 0.02425
    phigh = 1 - plow

    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
        )

    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
        )

    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (
        (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    )


def required_n_per_group(p1: float, p2: float, alpha: float, power: float) -> int:
    if not (0 < p1 < 1 and 0 < p2 < 1):
        raise ValueError("p1 and p2 must be in (0, 1)")

    delta = abs(p1 - p2)
    if delta <= 0:
        raise ValueError("p1 and p2 must differ")

    z_alpha = _normal_inv_cdf(1 - alpha / 2)
    z_beta = _normal_inv_cdf(power)

    p_bar = (p1 + p2) / 2
    term1 = z_alpha * math.sqrt(2 * p_bar * (1 - p_bar))
    term2 = z_beta * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))

    n = ((term1 + term2) ** 2) / (delta ** 2)
    return math.ceil(n)


def ci_half_width(p: float, n: int, alpha: float) -> float:
    z_alpha = _normal_inv_cdf(1 - alpha / 2)
    se = math.sqrt(p * (1 - p) / n)
    return z_alpha * se


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate sample size for significance and CI precision")
    parser.add_argument("--p1", type=float, required=True, help="Expected accuracy for system A (0-1)")
    parser.add_argument("--p2", type=float, required=True, help="Expected accuracy for system B (0-1)")
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance level (default 0.05)")
    parser.add_argument("--power", type=float, default=0.8, help="Target power (default 0.8)")
    parser.add_argument(
        "--ci-p",
        type=float,
        default=None,
        help="Optional expected accuracy for CI width planning; defaults to p1",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    n = required_n_per_group(args.p1, args.p2, args.alpha, args.power)
    ci_p = args.ci_p if args.ci_p is not None else args.p1
    ci_margin = ci_half_width(ci_p, n, args.alpha)

    print("Sample size planning result")
    print(f"- Inputs: p1={args.p1:.3f}, p2={args.p2:.3f}, alpha={args.alpha:.3f}, power={args.power:.3f}")
    print(f"- Required n per group (approx): {n}")
    print(f"- Approx 95% CI half-width at p={ci_p:.3f}: +/- {ci_margin:.3f}")
    print(f"- Approx total samples for two independent groups: {2 * n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
