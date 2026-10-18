"""
Two-Dimensional Two-Phase Flow Model (Chapter VI analog)
==========================================================

R. Marquez, "Modeling Downhole Natural Separation", PhD Dissertation,
The University of Tulsa, 2004 - Chapter VI.

IMPORTANT - SCOPE / FIDELITY NOTE
----------------------------------
Chapter VI solves the FULLY COUPLED two-phase mass + momentum balance
equations (gas AND liquid stream functions, pressure, and void fraction,
all solved simultaneously) via a custom finite-difference discretization
detailed across Appendix E (~25 pages), with drag/interfacial-area
closures in Appendices D and Chapter V. That exact numerical scheme is
not practical to reproduce line-for-line here.

What IS reproduced faithfully:
  * The governing equations actually shown in the dissertation text:
      - Eq. 6.1        mass balance (continuity)
      - Eqs. 6.4-6.5    stream function definition
      - Eq. 6.6-6.10    vorticity / stream-function relation
      - Eq. 4.35 / 4.54 Ishii terminal velocity & radial slip closure
                          (reused here exactly as in Ch. IV/V)
  * The same overall STRATEGY the chapter's introduction describes for
    the model's predecessor (Liu, 2001) and extends: solve the liquid
    flow field in the annulus/pump-intake domain first, then obtain the
    two-phase (gas) behavior via a force balance / trajectory-tracking
    scheme for individual bubbles moving through that field -- exactly
    the same physical picture as Chapter V, but now driven by a
    NUMERICALLY SOLVED liquid velocity field instead of an assumed
    streamline shape.

Concretely, this script:
  1. Solves the axisymmetric, inviscid, irrotational liquid stream
     function (Stokes stream function, E^2(psi) = 0) on a 2-D
     finite-difference grid spanning the annulus in front of the pump
     intake, with boundary conditions matching Fig. 6.2/6.3 (uniform
     inflow at the bottom, no-flow casing and pump walls, a suction
     velocity distributed across the pump intake ports).
  2. Recovers the liquid velocity field (V_lr, V_lz) from the stream
     function.
  3. Tracks individual gas bubble trajectories through that field using
     the SAME slip-velocity closures as Chapters IV/V (Ishii vertical
     terminal velocity + the liquid-radial-acceleration closure for
     radial slip), now evaluated LOCALLY from the numerical field.
  4. Classifies each bubble as "captured" (enters the pump) or "vented"
     (escapes to the annulus outlet) and computes natural separation
     efficiency E as the vented gas fraction.

This is a genuine, physically consistent 2-D numerical model built on
the dissertation's own equations and logic -- not a claim of exact
numerical equivalence to Marquez's specific finite-difference solution.
"""

from __future__ import annotations
from dataclasses import dataclass
from math import pi, sqrt
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
from scipy.interpolate import RegularGridInterpolator
from scipy.integrate import solve_ivp


# ---------------------------------------------------------------------
# Unit conversions (same convention as the simplified-model script)
# ---------------------------------------------------------------------
BBL_TO_M3       = 0.158987294928
DAY_TO_S        = 86400.0
IN_TO_M         = 0.0254
LBM_FT3_TO_KGM3 = 16.01846337
DYNE_CM_TO_N_M  = 0.001
M_S_TO_FT_S     = 3.280839895
G_SI            = 9.80665


@dataclass
class FluidProperties:
    rho_l_lbm_ft3: float
    rho_g_lbm_ft3: float
    sigma_dyne_cm: float

    def si(self):
        return (self.rho_l_lbm_ft3 * LBM_FT3_TO_KGM3,
                self.rho_g_lbm_ft3 * LBM_FT3_TO_KGM3,
                self.sigma_dyne_cm * DYNE_CM_TO_N_M)


@dataclass
class Geometry2D:
    """Bottomhole geometry for the 2-D domain (Fig. 6.2/6.3)."""
    casing_id_in: float        # r_c boundary
    pump_od_in: float          # r_p boundary
    intake_height_in: float    # h_p : vertical extent of the pump intake ports
    inlet_length_in: float     # length of annulus below the intake (flow development)
    A_p_in2: float | None = None  # pump intake port area; default = lateral area

    def r_p_m(self):  return 0.5 * self.pump_od_in * IN_TO_M
    def r_c_m(self):  return 0.5 * self.casing_id_in * IN_TO_M

    def A_a_m2(self):
        return pi * (self.r_c_m()**2 - self.r_p_m()**2)

    def A_p_m2(self):
        if self.A_p_in2 is not None:
            return self.A_p_in2 * IN_TO_M**2
        # default: lateral cylindrical area of the intake band
        return pi * (self.pump_od_in * IN_TO_M) * (self.intake_height_in * IN_TO_M)

    def z1_z2_m(self):
        z1 = self.inlet_length_in * IN_TO_M
        z2 = z1 + self.intake_height_in * IN_TO_M
        return z1, z2


# ---------------------------------------------------------------------
# 1) Liquid stream-function solve  (Eqs. 6.1, 6.4-6.6 -> E^2(psi)=0)
# ---------------------------------------------------------------------
class LiquidFlowField:
    """
    Axisymmetric, inviscid, irrotational liquid stream function on a
    rectangular (r,z) grid spanning [r_p, r_c] x [0, z2].

    Governing equation (Stokes stream function, zero vorticity):
        d2(psi)/dr2 - (1/r) d(psi)/dr + d2(psi)/dz2 = 0

    Velocities recovered per Eqs. 6.4-6.5 (single-phase form):
        V_r =  (1/r) d(psi)/dz
        V_z = -(1/r) d(psi)/dr
    """

    def __init__(self, geom: Geometry2D, V_sl: float,
                 Nr: int = 41, Nz: int = 81):
        self.geom = geom
        self.V_sl = V_sl
        self.r_p, self.r_c = geom.r_p_m(), geom.r_c_m()
        self.z1, self.z2 = geom.z1_z2_m()
        self.Nr, self.Nz = Nr, Nz

        self.r = np.linspace(self.r_p, self.r_c, Nr)
        self.z = np.linspace(0.0, self.z2, Nz)
        self.dr = self.r[1] - self.r[0]
        self.dz = self.z[1] - self.z[0]

        self.psi = self._solve()
        self.Vr, self.Vz = self._velocities()
        self._build_interpolators()

    # -- boundary values -------------------------------------------------
    def _psi_c(self):
        """Stream function value on the casing wall (total liquid flow)."""
        return 0.5 * self.V_sl * (self.r_p**2 - self.r_c**2)

    def _psi_inlet(self, r):
        """Uniform-inflow profile at z = 0 (Eq. matches V_z = V_sl)."""
        return 0.5 * self.V_sl * (self.r_p**2 - r**2)

    def _psi_pumpwall(self, z):
        """
        0 below the intake (solid wall, no radial flow), linear ramp
        across the intake band (distributed suction into the pump),
        reaching psi_c at the top of the intake (z2) -- matching the
        stagnant-liquid region assumed above the pump (Ch. IV/V).
        """
        psi_c = self._psi_c()
        if z <= self.z1:
            return 0.0
        elif z >= self.z2:
            return psi_c
        else:
            return psi_c * (z - self.z1) / (self.z2 - self.z1)

    # -- elliptic solve ----------------------------------------------------
    def _solve(self):
        Nr, Nz = self.Nr, self.Nz
        dr, dz = self.dr, self.dz
        N = Nr * Nz

        def idx(i, j):
            return i * Nz + j

        A = lil_matrix((N, N))
        b = np.zeros(N)
        psi_c = self._psi_c()

        for i in range(Nr):
            r_i = self.r[i]
            for j in range(Nz):
                k = idx(i, j)

                # Dirichlet boundaries
                if j == 0:                                   # inlet
                    A[k, k] = 1.0
                    b[k] = self._psi_inlet(r_i)
                elif j == Nz - 1:                             # top of domain
                    A[k, k] = 1.0
                    b[k] = psi_c
                elif i == Nr - 1:                             # casing wall
                    A[k, k] = 1.0
                    b[k] = psi_c
                elif i == 0:                                  # pump wall / intake
                    A[k, k] = 1.0
                    b[k] = self._psi_pumpwall(self.z[j])
                else:
                    # interior: d2psi/dr2 - (1/r) dpsi/dr + d2psi/dz2 = 0
                    c_e = 1.0 / dr**2 - 1.0 / (2 * r_i * dr)
                    c_w = 1.0 / dr**2 + 1.0 / (2 * r_i * dr)
                    c_n = 1.0 / dz**2
                    c_s = 1.0 / dz**2
                    c_p = -2.0 / dr**2 - 2.0 / dz**2

                    A[k, idx(i + 1, j)] = c_e
                    A[k, idx(i - 1, j)] = c_w
                    A[k, idx(i, j + 1)] = c_n
                    A[k, idx(i, j - 1)] = c_s
                    A[k, k] = c_p
                    b[k] = 0.0

        psi_flat = spsolve(csr_matrix(A), b)
        return psi_flat.reshape(Nr, Nz)

    def _velocities(self):
        Vr = np.zeros_like(self.psi)
        Vz = np.zeros_like(self.psi)
        for i in range(self.Nr):
            r_i = max(self.r[i], 1e-6)
            for j in range(self.Nz):
                # central differences (one-sided at edges)
                if 0 < j < self.Nz - 1:
                    dpsidz = (self.psi[i, j + 1] - self.psi[i, j - 1]) / (2 * self.dz)
                elif j == 0:
                    dpsidz = (self.psi[i, 1] - self.psi[i, 0]) / self.dz
                else:
                    dpsidz = (self.psi[i, -1] - self.psi[i, -2]) / self.dz

                if 0 < i < self.Nr - 1:
                    dpsidr = (self.psi[i + 1, j] - self.psi[i - 1, j]) / (2 * self.dr)
                elif i == 0:
                    dpsidr = (self.psi[1, j] - self.psi[0, j]) / self.dr
                else:
                    dpsidr = (self.psi[-1, j] - self.psi[-2, j]) / self.dr

                Vr[i, j] = dpsidz / r_i
                Vz[i, j] = -dpsidr / r_i
        return Vr, Vz

    def _build_interpolators(self):
        self._Vr_interp = RegularGridInterpolator(
            (self.r, self.z), self.Vr, bounds_error=False, fill_value=None)
        self._Vz_interp = RegularGridInterpolator(
            (self.r, self.z), self.Vz, bounds_error=False, fill_value=None)
        # radial gradient of Vr, needed for the radial-slip closure (Eq. 4.54)
        dVr_dr = np.gradient(self.Vr, self.r, axis=0)
        self._dVrdr_interp = RegularGridInterpolator(
            (self.r, self.z), dVr_dr, bounds_error=False, fill_value=None)

    def velocity(self, r, z):
        pt = np.array([[r, z]])
        return float(self._Vr_interp(pt)[0]), float(self._Vz_interp(pt)[0])

    def dVr_dr(self, r, z):
        pt = np.array([[r, z]])
        return float(self._dVrdr_interp(pt)[0])


# ---------------------------------------------------------------------
# 2) Gas-phase closures (reused exactly from Chapters IV/V)
# ---------------------------------------------------------------------
def terminal_velocity_ishii(fluids: FluidProperties) -> float:
    """Eq. 4.35 -- bubble terminal (rise) velocity, slug/churn flow."""
    rho_l, rho_g, sigma = fluids.si()
    return sqrt(2.0) * ((sigma * (rho_l - rho_g) * G_SI) / rho_l**2) ** 0.25


def radial_slip_ratio(V_lr: float, dVlr_dr: float,
                       fluids: FluidProperties) -> float:
    """
    Eq. 4.54 -- ratio of radial to vertical terminal (slip) velocity,
    driven by the local liquid radial acceleration:

        V_inf_r / V_inf_z = [ rho_l / ((rho_l - rho_g) g) ] * V_lr dV_lr/dr
    """
    rho_l, rho_g, _ = fluids.si()
    return (rho_l / ((rho_l - rho_g) * G_SI)) * V_lr * dVlr_dr


# ---------------------------------------------------------------------
# 3) Bubble trajectory tracking through the numerical liquid field
# ---------------------------------------------------------------------
class TwoDimensionalSeparationModel:
    """
    Ties together the liquid flow-field solve and bubble trajectory
    tracking to predict natural separation efficiency, E.
    """

    def __init__(self, geom: Geometry2D, fluids: FluidProperties,
                 q_l_bpd: float, q_g_bpd: float,
                 Nr: int = 41, Nz: int = 81, n_bubbles: int = 25):
        self.geom = geom
        self.fluids = fluids
        self.q_l_bpd = q_l_bpd
        self.q_g_bpd = q_g_bpd

        A_a = geom.A_a_m2()
        q_l_m3s = q_l_bpd * BBL_TO_M3 / DAY_TO_S
        self.V_sl = q_l_m3s / A_a

        self.field = LiquidFlowField(geom, self.V_sl, Nr=Nr, Nz=Nz)
        self.V_inf_z = terminal_velocity_ishii(fluids)
        self.n_bubbles = n_bubbles

    def _gas_velocity(self, r, z):
        V_lr, V_lz = self.field.velocity(r, z)
        dVlr_dr = self.field.dVr_dr(r, z)
        r_ratio = radial_slip_ratio(V_lr, dVlr_dr, self.fluids)
        V_inf_r = r_ratio * self.V_inf_z
        return V_lr + V_inf_r, V_lz + self.V_inf_z

    def _trajectory_rhs(self, t, y):
        r, z = y
        r = min(max(r, self.field.r_p), self.field.r_c)
        z = min(max(z, 0.0), self.field.z2)
        Vgr, Vgz = self._gas_velocity(r, z)
        return [Vgr, Vgz]

    def _hit_pump(self, t, y):
        return y[0] - self.field.r_p
    _hit_pump.terminal = True
    _hit_pump.direction = -1

    def _hit_top(self, t, y):
        return y[1] - self.field.z2
    _hit_top.terminal = True
    _hit_top.direction = 1

    def run(self):
        r_p, r_c = self.field.r_p, self.field.r_c
        # inlet positions weighted by annular area (equal-flux increments)
        r2 = np.linspace(r_p**2, r_c**2, self.n_bubbles + 1)
        r_mid = np.sqrt(0.5 * (r2[:-1] + r2[1:]))
        weights = np.diff(r2)             # proportional to annular area
        weights = weights / weights.sum()

        vented_weight = 0.0
        t_span = (0.0, 200.0)              # generous upper time bound
        self.trajectories = []             # (r0, weight, vented, r_path, z_path) in m

        for r0, w in zip(r_mid, weights):
            sol = solve_ivp(self._trajectory_rhs, t_span, [r0, 0.0],
                             events=[self._hit_pump, self._hit_top],
                             max_step=0.02, rtol=1e-6, atol=1e-9)
            vented = sol.t_events[1].size > 0
            self.trajectories.append((r0, w, vented, sol.y[0], sol.y[1]))
            if vented:                         # reached the top -> vented
                vented_weight += w
            # else: captured by the pump (or ran out of time -> treat as captured)

        E = vented_weight
        q_g_vented = self.q_g_bpd * E
        q_g_pump = self.q_g_bpd - q_g_vented
        return {
            "E": E,
            "V_sl_ft_s": self.V_sl * M_S_TO_FT_S,
            "V_inf_ft_s": self.V_inf_z * M_S_TO_FT_S,
            "q_g_pump_bpd": q_g_pump,
            "q_g_vented_bpd": q_g_vented,
        }


# ---------------------------------------------------------------------
# Example run, using the validation case quoted in the dissertation
# (Ch. VII, "Accuracy of the Model"): 6.4-in casing, 4-in pump,
# 150 psi, 75 F, GLR 5 scf/bbl, 1500 bpd liquid, air-water, bubble flow.
# ---------------------------------------------------------------------
if __name__ == "__main__":

    geom = Geometry2D(
        casing_id_in=6.4,
        pump_od_in=4.0,
        intake_height_in=3.0,      # pump intake port height
        inlet_length_in=8.0,       # annulus development length below intake
    )

    fluids = FluidProperties(
        rho_l_lbm_ft3=62.0,        # water at ~150 psi, 75F
        rho_g_lbm_ft3=1.0,         # air at local conditions (approx.)
        sigma_dyne_cm=60.0,        # air-water interfacial tension
    )

    model = TwoDimensionalSeparationModel(
        geom, fluids, q_l_bpd=1500.0, q_g_bpd=650.0,   # GLR 5 scf/bbl (approx local)
        Nr=41, Nz=81, n_bubbles=25,
    )

    result = model.run()

    print("=" * 56)
    print("2-D TWO-PHASE FLOW MODEL (Chapter VI analog)")
    print("=" * 56)
    for k, v in result.items():
        print(f"  {k:16s}: {v:10.4f}")
    print("=" * 56)
