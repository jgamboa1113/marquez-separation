"""
Fit a Maximum-Likelihood surrogate to the 2-D Two-Phase Flow Model
====================================================================

Workflow:
  1. Run `TwoDimensionalSeparationModel` (two_dimensional_model.py) across
     a sweep of conditions modeled on Marquez's own sensitivity studies
     (Chapter VII): casing size (5, 6, 8 in, per Fig. 4.1), liquid flow
     rate, and gas-liquid ratio.
  2. Collect (inputs -> E) as a synthetic dataset, playing the same role
     as his TUALP/Alhanati/Serrano experimental points did for the
     Chapter IV "New Correlation".
  3. Fit a compact closed-form surrogate to that dataset via Maximum
     Likelihood Estimation (Gaussian residual model), so the expensive
     2-D numerical solve can be replaced by a fast algebraic formula --
     exactly the same idea behind Marquez's own empirical M-parameter
     correlation (Eq. 4.59), just re-derived here against the 2-D
     model's own output instead of physical experiments.

Surrogate form
--------------
Two dimensionless groups drive the physics:
    x = V_sl / V_inf_z      (liquid superficial velocity / bubble rise velocity)
    g = A_p  / A_a          (pump-intake-port area / annulus area)

A logistic form guarantees 0 <= E <= 1 by construction:

    E_hat(x, g; theta) = 1 / (1 + exp[ t0 + t1*x + t2*g + t3*x*g + t4*x^2 ])

Assuming E_2D = E_hat(x,g;theta) + eps,  eps ~ N(0, sigma^2) iid, the
log-likelihood is:

    ln L(theta,sigma) = -(n/2) ln(2*pi*sigma^2) - (1/(2*sigma^2)) * SSE(theta)

Maximizing ln L over theta is equivalent to minimizing the sum of
squared errors SSE(theta); sigma_hat^2 = SSE(theta_hat)/n follows in
closed form. Both are estimated here explicitly via numerical
optimization of the negative log-likelihood (rather than silently
calling a least-squares routine) so the MLE machinery is visible.
"""

import numpy as np
from scipy.optimize import minimize

from two_dimensional_model import (
    Geometry2D, FluidProperties, TwoDimensionalSeparationModel,
    terminal_velocity_ishii,
)


# ---------------------------------------------------------------------
# 1) Build the synthetic dataset by sweeping conditions
# ---------------------------------------------------------------------
def generate_dataset():
    fluids = FluidProperties(
        rho_l_lbm_ft3=62.0,
        rho_g_lbm_ft3=1.0,
        sigma_dyne_cm=60.0,
    )

    casing_sizes_in = [5.0, 6.0, 8.0]        # Fig. 4.1 "Radial Drag Effect" cases
    pump_od_in = 4.0
    liquid_rates_bpd = [300, 600, 1000, 1500, 2000, 2500, 3000]
    glr_scf_bbl = [3.0, 5.0, 8.0]            # local gas-liquid ratio variants

    rows = []
    for D_c in casing_sizes_in:
        geom = Geometry2D(
            casing_id_in=D_c, pump_od_in=pump_od_in,
            intake_height_in=3.0, inlet_length_in=8.0,
        )
        A_p, A_a = geom.A_p_m2(), geom.A_a_m2()
        g_ratio = A_p / A_a
        V_inf = terminal_velocity_ishii(fluids)

        for q_l in liquid_rates_bpd:
            for glr in glr_scf_bbl:
                q_g = q_l * glr / 5.615     # crude scf/bbl -> bbl/d-equivalent scaling

                model = TwoDimensionalSeparationModel(
                    geom, fluids, q_l_bpd=q_l, q_g_bpd=q_g,
                    Nr=31, Nz=61, n_bubbles=21,
                )
                result = model.run()

                V_sl_m_s = result["V_sl_ft_s"] / 3.280839895
                x = V_sl_m_s / V_inf

                rows.append(dict(
                    casing_id_in=D_c, q_l_bpd=q_l, glr=glr,
                    x=x, g=g_ratio, E=result["E"],
                ))
                print(f"  D_c={D_c:4.1f} in | q_l={q_l:5.0f} bpd | "
                      f"GLR={glr:4.1f} | x={x:6.3f} | g={g_ratio:6.3f} "
                      f"| E={result['E']:.4f}")

    return rows


# ---------------------------------------------------------------------
# 2) Surrogate model + MLE fitting
# ---------------------------------------------------------------------
def logistic_surrogate(x, g, theta):
    t0, t1, t2, t3, t4 = theta
    z = t0 + t1 * x + t2 * g + t3 * x * g + t4 * x**2
    z = np.clip(z, -50, 50)          # avoid overflow in exp
    return 1.0 / (1.0 + np.exp(z))


def neg_log_likelihood(params, x, g, E_obs):
    theta = params[:5]
    log_sigma = params[5]
    sigma = np.exp(log_sigma)

    E_hat = logistic_surrogate(x, g, theta)
    resid = E_obs - E_hat
    n = len(E_obs)

    nll = 0.5 * n * np.log(2 * np.pi * sigma**2) + 0.5 * np.sum(resid**2) / sigma**2
    return nll


def fit_mle(rows):
    x = np.array([r["x"] for r in rows])
    g = np.array([r["g"] for r in rows])
    E = np.array([r["E"] for r in rows])

    theta0 = np.array([0.5, -1.5, 0.0, 0.0, 0.0])
    log_sigma0 = np.log(0.1)
    params0 = np.concatenate([theta0, [log_sigma0]])

    res = minimize(neg_log_likelihood, params0, args=(x, g, E),
                    method="Nelder-Mead",
                    options=dict(maxiter=20000, xatol=1e-8, fatol=1e-10))

    theta_hat = res.x[:5]
    sigma_hat = np.exp(res.x[5])

    E_hat = logistic_surrogate(x, g, theta_hat)
    resid = E - E_hat
    sse = np.sum(resid**2)
    sst = np.sum((E - E.mean())**2)
    r2 = 1 - sse / sst

    return dict(theta=theta_hat, sigma=sigma_hat, r2=r2,
                converged=res.success, nll=res.fun,
                x=x, g=g, E=E, E_hat=E_hat)


# ---------------------------------------------------------------------
# Run everything
# ---------------------------------------------------------------------
if __name__ == "__main__":

    print("=" * 70)
    print("STEP 1: Running the 2-D model across a sweep of conditions")
    print("=" * 70)
    rows = generate_dataset()

    print()
    print("=" * 70)
    print("STEP 2: Fitting the MLE surrogate")
    print("=" * 70)
    fit = fit_mle(rows)

    names = ["theta0", "theta1 (x)", "theta2 (g)", "theta3 (x*g)", "theta4 (x^2)"]
    print(f"Converged        : {fit['converged']}")
    print(f"Neg. log-lik.    : {fit['nll']:.4f}")
    print(f"Fitted sigma     : {fit['sigma']:.5f}")
    print(f"R^2 (vs 2D model): {fit['r2']:.4f}")
    print("-" * 70)
    for name, val in zip(names, fit["theta"]):
        print(f"  {name:14s} = {val:+9.4f}")
    print("-" * 70)

    print("\nSample comparison (2-D model vs. MLE surrogate):")
    print(f"{'x':>8s} {'g':>8s} {'E_2D':>8s} {'E_MLE':>8s} {'resid':>8s}")
    idx = np.linspace(0, len(fit["E"]) - 1, 12).astype(int)
    for i in idx:
        print(f"{fit['x'][i]:8.3f} {fit['g'][i]:8.4f} "
              f"{fit['E'][i]:8.4f} {fit['E_hat'][i]:8.4f} "
              f"{fit['E'][i]-fit['E_hat'][i]:+8.4f}")
