"""
Downhole Natural Separation Efficiency - Three Models
=======================================================

R. Marquez, "Modeling Downhole Natural Separation", PhD Dissertation,
The University of Tulsa, 2004.

  1) Alhanati (1993) simplified model ............. Eq. 4.38   (Ch. IV)
  2) Marquez new simplified model (M correlation) .. Eq. 4.57/4.59 (Ch. IV)
  3) Marquez mechanistic model (bubble tracking) ... Eqs. 5.1-5.45 (Ch. V)

Common inputs (field units)
---------------------------
  q_l_bpd, q_g_bpd : liquid / gas flow rates at LOCAL (downhole) conditions, bbl/d
  casing_id_in     : casing inside diameter, in
  pump_od_in       : pump intake / motor outside diameter, in
  rho_l, rho_g     : local liquid / gas density, lbm/ft3
  sigma            : gas-liquid interfacial tension, dyne/cm
Additional inputs for the mechanistic model only
  intake_height_in : pump intake port height h_p, in
  mu_l_cp          : liquid viscosity, cP

Mechanistic model recap (Chapter V)
-----------------------------------
Liquid streamlines run vertically, then turn radially toward the intake
along a straight line at angle beta from the vertical,
    tan(beta) = (r_c - r_p) / h_p                                (Eq. 5.3)
A streamline starting at inlet radius r_i has radial velocity
    V_lr = -(r_i / r) tan(beta) V_lz                            (Eq. 5.9)
Gas bubbles feel Stokes drag + a pressure-gradient force from the liquid's
radial acceleration, which gives (Eq. 5.42, with l_i replacing r_d):

    dr/dz = -54 mu_l / (l_i^2 (rho_l - rho_g) g) * A *
            [ (2 l_i^2 rho_l / (9 mu_l)) * A / r^3 + 1 / r ],
    A = r_i tan(beta) V_lz                                       

A bubble enters the radial region at height h_i (where the diagonal
streamline-turning line meets r = r_i) and is captured if it reaches
r = r_p before the top of the intake (z = h_p). The inlet radius that just
makes it defines the "separation circle" r_s and (Eq. 5.41)

    E = (r_c^2 - r_s^2) / (r_c^2 - r_p^2)

l_i (interface characteristic length) comes from the TUALP-fitted
interfacial-area correlation (Eqs. 5.43-5.45 / 7.8).
"""

from __future__ import annotations
from dataclasses import dataclass
from math import pi, sqrt, log, exp, atan

from scipy.integrate import solve_ivp
from scipy.optimize import brentq

# ---------------------------------------------------------------------
# Unit conversions
# ---------------------------------------------------------------------
BBL_TO_M3       = 0.158987294928
DAY_TO_S        = 86400.0
IN_TO_M         = 0.0254
LBM_FT3_TO_KGM3 = 16.01846337
DYNE_CM_TO_N_M  = 0.001
CP_TO_PA_S      = 0.001
M_S_TO_FT_S     = 3.280839895
G_SI            = 9.80665


# ---------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------
@dataclass
class WellConditions:
    q_l_bpd: float            # liquid rate, local conditions
    q_g_bpd: float            # gas rate, local conditions
    casing_id_in: float
    pump_od_in: float
    rho_l_lbm_ft3: float = 62.4
    rho_g_lbm_ft3: float = 1.0
    sigma_dyne_cm: float = 60.0
    mu_l_cp: float = 1.0
    intake_height_in: float = 3.0     # needed by the mechanistic model

    # ---- SI helpers ----
    @property
    def r_c(self): return 0.5 * self.casing_id_in * IN_TO_M
    @property
    def r_p(self): return 0.5 * self.pump_od_in * IN_TO_M
    @property
    def h_p(self): return self.intake_height_in * IN_TO_M
    @property
    def rho_l(self): return self.rho_l_lbm_ft3 * LBM_FT3_TO_KGM3
    @property
    def rho_g(self): return self.rho_g_lbm_ft3 * LBM_FT3_TO_KGM3
    @property
    def sigma(self): return self.sigma_dyne_cm * DYNE_CM_TO_N_M
    @property
    def mu_l(self): return self.mu_l_cp * CP_TO_PA_S

    @property
    def A_a(self):
        """Annulus flow area, m2."""
        return pi * (self.r_c**2 - self.r_p**2)

    @property
    def V_sl(self):
        """Superficial liquid velocity in the annulus, m/s (Eq. 5.12)."""
        return self.q_l_bpd * BBL_TO_M3 / DAY_TO_S / self.A_a

    @property
    def V_sg(self):
        """Superficial gas velocity in the annulus, m/s."""
        return self.q_g_bpd * BBL_TO_M3 / DAY_TO_S / self.A_a


# ---------------------------------------------------------------------
# Shared: bubble terminal velocity (Ishii, Eq. 4.35 / 7.2)
# ---------------------------------------------------------------------
def terminal_velocity_ishii(w: WellConditions) -> float:
    return sqrt(2.0) * ((w.sigma * (w.rho_l - w.rho_g) * G_SI) / w.rho_l**2) ** 0.25


# ---------------------------------------------------------------------
# Model 1: Alhanati (Eq. 4.38)
# ---------------------------------------------------------------------
def alhanati_efficiency(w: WellConditions) -> float:
    """E = V_inf / (V_sl + V_inf)"""
    V_inf = terminal_velocity_ishii(w)
    return V_inf / (w.V_sl + V_inf)


# ---------------------------------------------------------------------
# Model 2: Marquez simplified, M-parameter correlation (Eqs. 4.57, 4.59)
# ---------------------------------------------------------------------
MQ_A, MQ_B, MQ_C, MQ_D = -0.0093, 57.758, 34.40, 1.308


def marquez_simplified_efficiency(w: WellConditions) -> float:
    """
    x = V_sl / V_inf ;  M = -(a b + c x^d)/(b + x^d) ;  E = 1 - (M + x)
    Clipped to [0, 1] (equivalent to the M1/M2 bounds, Eqs. 4.60-4.61).
    """
    V_inf = terminal_velocity_ishii(w)
    x = w.V_sl / V_inf
    M = -((MQ_A * MQ_B + MQ_C * x**MQ_D) / (MQ_B + x**MQ_D))
    E = 1.0 - (M + x)
    return min(max(E, 0.0), 1.0)


# ---------------------------------------------------------------------
# Model 3: Marquez mechanistic model (Chapter V)
# ---------------------------------------------------------------------
def interface_length_m(w: WellConditions) -> float:
    """
    Interface characteristic length l_i (Eq. 5.45 / 7.8):

        l_i = 3 / [71.617 (1 - exp(-2.368 V_sgz))] * V_sgz / (V_sgz + V_inf)

    The correlation was fitted with V_sgz in ft/s, a_i in 1/in, so l_i
    comes out in inches here; it is converted to metres on return.
    """
    V_sg_ft = w.V_sg * M_S_TO_FT_S
    V_inf_ft = terminal_velocity_ishii(w) * M_S_TO_FT_S

    if V_sg_ft < 1e-9:
        # limit V_sg -> 0 of Eq. 5.45:  l_i -> 3 / (71.617*2.368*(V_sg+V_inf))
        l_in = 3.0 / (71.617 * 2.368 * V_inf_ft)
    else:
        a_i = 71.617 * (1.0 - exp(-2.368 * V_sg_ft))         # Eq. 5.44, 1/in
        alpha = V_sg_ft / (V_sg_ft + V_inf_ft)
        l_in = 3.0 * alpha / a_i                              # Eq. 5.43
    return l_in * IN_TO_M


def _trajectory_constants(w: WellConditions, l_i: float):
    """K1, K2 such that dr/dz = -K1*A*(K2*A/r^3 + 1/r)   (Eq. 5.42)."""
    K1 = 54.0 * w.mu_l / (l_i**2 * (w.rho_l - w.rho_g) * G_SI)
    K2 = 2.0 * l_i**2 * w.rho_l / (9.0 * w.mu_l)
    return K1, K2


def _streamline_params(w: WellConditions, r_i: float):
    """h_i (Eq. 5.10 inverted) and A = r_i tan(beta) V_lz (Eqs. 5.9, 5.12)."""
    tan_beta = (w.r_c - w.r_p) / w.h_p                        # Eq. 5.3
    h_i = w.h_p * (r_i - w.r_p) / (w.r_c - w.r_p)
    A = r_i * tan_beta * w.V_sl
    return h_i, A


def vertical_rise_to_pump_analytic(w, r_i, l_i):
    """
    Closed-form integral of Eq. 5.42: vertical distance a bubble needs
    to travel (from r = r_i) to reach r = r_p.

        dz = - r^3 dr / (K1 A (K2 A + r^2))
        dz_total = 1/(K1 A) [ (r_i^2 - r_p^2)/2
                              - (K2 A / 2) ln((K2 A + r_i^2)/(K2 A + r_p^2)) ]
    """
    K1, K2 = _trajectory_constants(w, l_i)
    _, A = _streamline_params(w, r_i)
    term1 = 0.5 * (r_i**2 - w.r_p**2)
    term2 = 0.5 * K2 * A * log((K2 * A + r_i**2) / (K2 * A + w.r_p**2))
    return (term1 - term2) / (K1 * A)


def bubble_trajectory_numeric(w, r_i, l_i, n_pts=400):
    """
    Numerically integrate Eq. 5.42 from (r_i, h_i) upward until the bubble
    either hits the pump (r = r_p) or leaves the top of the intake (z = h_p).
    Returns (captured: bool, z_array, r_array).
    """
    K1, K2 = _trajectory_constants(w, l_i)
    h_i, A = _streamline_params(w, r_i)

    def rhs(z, y):
        r = max(y[0], w.r_p * 0.5)
        return [-K1 * A * (K2 * A / r**3 + 1.0 / r)]

    def hit_pump(z, y):
        return y[0] - w.r_p
    hit_pump.terminal = True
    hit_pump.direction = -1

    sol = solve_ivp(rhs, (h_i, w.h_p), [r_i], events=hit_pump,
                    rtol=1e-9, atol=1e-12, dense_output=False,
                    max_step=(w.h_p / n_pts))
    captured = sol.t_events[0].size > 0
    return captured, sol.t, sol.y[0]


def separation_radius(w: WellConditions, l_i: float, method="analytic") -> float:
    """
    Find the separation-circle radius r_s: the inlet radius whose bubble just
    reaches the upper edge of the pump intake (r = r_p at z = h_p).
    Bubbles starting at r_i < r_s are captured, r_i > r_s are vented.
    """
    def f_analytic(r_i):
        h_i, _ = _streamline_params(w, r_i)
        return h_i + vertical_rise_to_pump_analytic(w, r_i, l_i) - w.h_p

    def f_numeric(r_i):
        captured, _, _ = bubble_trajectory_numeric(w, r_i, l_i)
        return -1.0 if captured else 1.0

    eps = 1e-9 * w.r_c
    lo, hi = w.r_p + eps, w.r_c
    if method == "analytic":
        f = f_analytic
        if f(hi) <= 0.0:          # even the outermost bubble is captured
            return w.r_c
        return brentq(f, lo, hi, xtol=1e-12)
    else:
        # bisection on the numerical capture predicate
        if f_numeric(hi) < 0.0:
            return w.r_c
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if f_numeric(mid) < 0.0:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)


def marquez_mechanistic_efficiency(w: WellConditions, method="analytic",
                                    return_details=False):
    """
    Natural separation efficiency from the mechanistic model, Eq. 5.41:
        E = (r_c^2 - r_s^2) / (r_c^2 - r_p^2)
    method = "analytic" (closed-form trajectory integral) or
             "numeric"  (solve_ivp on Eq. 5.42, bisection on r_s).
    """
    l_i = interface_length_m(w)
    r_s = separation_radius(w, l_i, method=method)
    E = (w.r_c**2 - r_s**2) / (w.r_c**2 - w.r_p**2)
    E = min(max(E, 0.0), 1.0)

    if return_details:
        K1, K2 = _trajectory_constants(w, l_i)
        V_gz = l_i**2 * (w.rho_l - w.rho_g) * G_SI / (54.0 * w.mu_l)  # Eq. 5.23, Stokes
        return dict(E=E, r_s_in=r_s / IN_TO_M, l_i_in=l_i / IN_TO_M,
                    V_gz_ft_s=V_gz * M_S_TO_FT_S,
                    tan_beta=(w.r_c - w.r_p) / w.h_p,
                    beta_deg=atan((w.r_c - w.r_p) / w.h_p) * 180.0 / pi)
    return E


# ---------------------------------------------------------------------
# Driver: run all three models and report
# ---------------------------------------------------------------------
def run_all(w: WellConditions) -> dict:
    E_alh = alhanati_efficiency(w)
    E_msi = marquez_simplified_efficiency(w)
    mech = marquez_mechanistic_efficiency(w, return_details=True)
    out = dict(
        V_sl_ft_s=w.V_sl * M_S_TO_FT_S,
        V_sg_ft_s=w.V_sg * M_S_TO_FT_S,
        V_inf_ft_s=terminal_velocity_ishii(w) * M_S_TO_FT_S,
        E_alhanati=E_alh,
        E_marquez_simplified=E_msi,
        E_marquez_mechanistic=mech["E"],
        mech_details=mech,
    )
    return out


def print_report(w: WellConditions) -> None:
    r = run_all(w)
    d = r["mech_details"]
    print("=" * 62)
    print("NATURAL SEPARATION EFFICIENCY - THREE-MODEL COMPARISON")
    print("=" * 62)
    print(f"q_l = {w.q_l_bpd:.0f} bpd | q_g = {w.q_g_bpd:.0f} bpd (local) | "
          f"casing ID = {w.casing_id_in} in | pump OD = {w.pump_od_in} in")
    print(f"intake height = {w.intake_height_in} in | mu_l = {w.mu_l_cp} cP")
    print("-" * 62)
    print(f"V_sl  = {r['V_sl_ft_s']:7.4f} ft/s   V_sg = {r['V_sg_ft_s']:7.4f} ft/s"
          f"   V_inf = {r['V_inf_ft_s']:6.4f} ft/s")
    print("-" * 62)
    print(f"  Alhanati (1993)               E = {r['E_alhanati']:.4f}")
    print(f"  Marquez simplified            E = {r['E_marquez_simplified']:.4f}")
    print(f"  Marquez mechanistic           E = {r['E_marquez_mechanistic']:.4f}")
    print("-" * 62)
    print("  Mechanistic details:")
    print(f"    streamline angle beta = {d['beta_deg']:.1f} deg "
          f"(tan = {d['tan_beta']:.3f})")
    print(f"    interface length l_i  = {d['l_i_in']:.5f} in")
    print(f"    bubble rise V_gz      = {d['V_gz_ft_s']:.4f} ft/s")
    print(f"    separation radius r_s = {d['r_s_in']:.4f} in "
          f"(r_p = {0.5*w.pump_od_in:.3f}, r_c = {0.5*w.casing_id_in:.3f})")
    print("=" * 62)


# ---------------------------------------------------------------------
# Example + self-check
# ---------------------------------------------------------------------
if __name__ == "__main__":

    # Validation-style case quoted in Ch. VII: 6.4-in casing, 4-in pump,
    # 3-in intake, air/water, 1500 bpd liquid.
    w = WellConditions(
        q_l_bpd=1500.0,
        q_g_bpd=500.0,
        casing_id_in=6.4,
        pump_od_in=4.0,
        rho_l_lbm_ft3=62.4,
        rho_g_lbm_ft3=0.8,
        sigma_dyne_cm=60.0,
        mu_l_cp=1.0,
        intake_height_in=3.0,
    )
    print_report(w)

    # Self-check: analytic trajectory integral vs. numerical ODE solution
    E_an = marquez_mechanistic_efficiency(w, method="analytic")
    E_nu = marquez_mechanistic_efficiency(w, method="numeric")
    print(f"\nSelf-check, mechanistic E:  analytic = {E_an:.5f} | "
          f"numeric (solve_ivp) = {E_nu:.5f} | diff = {abs(E_an - E_nu):.2e}")

    # Sweep liquid rate for all three models
    print("\nLiquid-rate sweep (same geometry/fluids, q_g = 500 bpd local)")
    print(f"{'q_l (bpd)':>10s} {'Alhanati':>10s} {'Mq simpl.':>10s} {'Mq mech.':>10s}")
    for q_l in (300, 600, 1000, 1500, 2000, 2500, 3000):
        w.q_l_bpd = float(q_l)
        print(f"{q_l:10d} {alhanati_efficiency(w):10.4f} "
              f"{marquez_simplified_efficiency(w):10.4f} "
              f"{marquez_mechanistic_efficiency(w):10.4f}")
