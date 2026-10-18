"""Interactive comparison of the 2-D separation model and its surrogate."""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from fit_surrogate_mle import fit_mle, logistic_surrogate
import natural_separation_all_models as nsm
from two_dimensional_model import (
    FluidProperties,
    Geometry2D,
    M_S_TO_FT_S,
    TwoDimensionalSeparationModel,
    terminal_velocity_ishii,
)


LIQUID_RATES_BPD = [300, 600, 1000, 1500, 2000, 2500, 3000]
CASING_SIZES_IN = [5.0, 6.0, 8.0]
PUMP_OD_IN = 4.0
INTAKE_HEIGHT_IN = 3.0
INLET_LENGTH_IN = 8.0
FLUIDS = FluidProperties(
    rho_l_lbm_ft3=62.0,
    rho_g_lbm_ft3=1.0,
    sigma_dyne_cm=60.0,
)


@st.cache_data(show_spinner=False)
def train_surrogate(nr: int, nz: int, n_bubbles: int):
    """Fit on unique casing/rate cases; the current 2-D model ignores gas rate."""
    rows = []
    v_inf = terminal_velocity_ishii(FLUIDS)

    for casing_id in CASING_SIZES_IN:
        geometry = Geometry2D(
            casing_id_in=casing_id,
            pump_od_in=PUMP_OD_IN,
            intake_height_in=INTAKE_HEIGHT_IN,
            inlet_length_in=INLET_LENGTH_IN,
        )
        g = geometry.A_p_m2() / geometry.A_a_m2()

        for liquid_rate in LIQUID_RATES_BPD:
            model = TwoDimensionalSeparationModel(
                geometry,
                FLUIDS,
                q_l_bpd=liquid_rate,
                q_g_bpd=0.0,
                Nr=nr,
                Nz=nz,
                n_bubbles=n_bubbles,
            )
            result = model.run()
            x = (result["V_sl_ft_s"] / M_S_TO_FT_S) / v_inf
            rows.append(
                {
                    "casing_id_in": casing_id,
                    "q_l_bpd": liquid_rate,
                    "x": x,
                    "g": g,
                    "E": result["E"],
                }
            )

    return rows, fit_mle(rows)


def gas_pvt(pressure_psia: float, temp_f: float, gas_sg: float, z_factor: float) -> dict:
    """Real-gas Bg (bbl/scf) and density (lbm/ft3) at intake conditions."""
    t_r = temp_f + 459.67
    bg_bbl_scf = 0.02827 * z_factor * t_r / pressure_psia / 5.615
    rho_g = 28.97 * gas_sg * pressure_psia / (10.732 * z_factor * t_r)
    return {"bg_bbl_scf": bg_bbl_scf, "rho_g": rho_g}


def local_gas_rate_bpd(liquid_rate: float, glr_scf_stb: float, pvt: dict) -> float:
    """Free gas at intake conditions: q_l * GLR * Bg (dissolved gas not modeled)."""
    return liquid_rate * glr_scf_stb * pvt["bg_bbl_scf"]


def published_models(casing_id: float, liquid_rate: float, gas_rate: float, rho_g: float) -> dict:
    """Alhanati, Marquez simplified and Marquez mechanistic efficiencies."""
    w = nsm.WellConditions(
        q_l_bpd=float(liquid_rate),
        q_g_bpd=float(gas_rate),
        casing_id_in=casing_id,
        pump_od_in=PUMP_OD_IN,
        rho_l_lbm_ft3=FLUIDS.rho_l_lbm_ft3,
        rho_g_lbm_ft3=rho_g,
        sigma_dyne_cm=FLUIDS.sigma_dyne_cm,
        mu_l_cp=1.0,
        intake_height_in=INTAKE_HEIGHT_IN,
    )
    return {
        "Alhanati": nsm.alhanati_efficiency(w),
        "Marquez simplified": nsm.marquez_simplified_efficiency(w),
        "Marquez mechanistic": nsm.marquez_mechanistic_efficiency(w),
    }


def plot_model_sweep(bundle, casing_id: float, liquid_rate: float, glr: float):
    rates = np.arange(200, 3201, 50)
    pvt = bundle["pvt"]
    curves = {"Alhanati": [], "Marquez simplified": [], "Marquez mechanistic": []}
    for q in rates:
        values = published_models(
            casing_id, q, local_gas_rate_bpd(q, glr, pvt), pvt["rho_g"]
        )
        for name, value in values.items():
            curves[name].append(value)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    styles = {
        "Alhanati": ("tab:green", "--"),
        "Marquez simplified": ("tab:orange", "-."),
        "Marquez mechanistic": ("tab:red", "-"),
    }
    for name, values in curves.items():
        color, ls = styles[name]
        ax.plot(rates, values, color=color, linestyle=ls, label=name)

    group = pd.DataFrame(bundle["rows"])
    group = group[group["casing_id_in"] == casing_id]
    if not group.empty:
        ax.scatter(group["q_l_bpd"], group["E"], color="black", s=30, label="2-D model (training)")
        surrogate = logistic_surrogate(
            group["x"].to_numpy(), group["g"].to_numpy(), bundle["fit"]["theta"]
        )
        ax.plot(group["q_l_bpd"], surrogate, color="tab:blue", label="Surrogate")
    ax.scatter(
        [liquid_rate], [bundle["result"]["E"]], marker="*", s=180,
        color="black", edgecolor="white", zorder=5, label="Selected case (2-D)",
    )
    ax.set(xlabel="Liquid rate (bbl/day)", ylabel="Separation efficiency E",
           title=f"All models vs liquid rate ({casing_id:g}-in casing, 4-in pump)")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    return fig


def run_case(
    casing_id: float,
    liquid_rate: float,
    glr: float,
    pressure_psia: float,
    temp_f: float,
    gas_sg: float,
    z_factor: float,
    nr: int,
    nz: int,
    n_bubbles: int,
):
    rows, fit = train_surrogate(nr, nz, n_bubbles)
    geometry = Geometry2D(
        casing_id_in=casing_id,
        pump_od_in=PUMP_OD_IN,
        intake_height_in=INTAKE_HEIGHT_IN,
        inlet_length_in=INLET_LENGTH_IN,
    )
    pvt = gas_pvt(pressure_psia, temp_f, gas_sg, z_factor)
    gas_rate = local_gas_rate_bpd(liquid_rate, glr, pvt)
    model = TwoDimensionalSeparationModel(
        geometry,
        FLUIDS,
        q_l_bpd=liquid_rate,
        q_g_bpd=gas_rate,
        Nr=nr,
        Nz=nz,
        n_bubbles=n_bubbles,
    )
    result = model.run()
    x = (result["V_sl_ft_s"] / M_S_TO_FT_S) / model.V_inf_z
    g = geometry.A_p_m2() / geometry.A_a_m2()
    surrogate_e = float(logistic_surrogate(np.array([x]), np.array([g]), fit["theta"])[0])

    field = model.field
    z_in = field.z / 0.0254
    r_in = field.r / 0.0254
    z_grid, r_grid = np.meshgrid(z_in, r_in)
    radial_ft_s = field.Vr * M_S_TO_FT_S
    axial_ft_s = field.Vz * M_S_TO_FT_S

    return {
        "rows": rows,
        "fit": fit,
        "result": result,
        "surrogate_e": surrogate_e,
        "published": published_models(casing_id, liquid_rate, gas_rate, pvt["rho_g"]),
        "pvt": pvt,
        "gas_rate_bpd": gas_rate,
        "trajectories": [
            (r0 / 0.0254, w, vented, r / 0.0254, z / 0.0254)
            for r0, w, vented, r, z in model.trajectories
        ],
        "r_p_in": 0.5 * PUMP_OD_IN,
        "r_c_in": 0.5 * casing_id,
        "x": x,
        "g": g,
        "z_grid": z_grid,
        "r_grid": r_grid,
        "speed_ft_s": np.hypot(radial_ft_s, axial_ft_s),
        "radial_ft_s": radial_ft_s,
        "axial_ft_s": axial_ft_s,
        "z1_in": INLET_LENGTH_IN,
        "z2_in": INLET_LENGTH_IN + INTAKE_HEIGHT_IN,
    }


def plot_mesh_contour(bundle, contour_choice: str):
    field_key = {
        "Liquid speed magnitude": "speed_ft_s",
        "Axial liquid velocity": "axial_ft_s",
        "Radial liquid velocity": "radial_ft_s",
    }[contour_choice]
    units = "ft/s"
    values = bundle[field_key]
    z_grid = bundle["z_grid"]
    r_grid = bundle["r_grid"]

    fig, ax = plt.subplots(figsize=(9, 5.2), constrained_layout=True)
    levels = np.linspace(float(np.nanmin(values)), float(np.nanmax(values)), 24)
    if levels[0] == levels[-1]:
        levels = 24
    contour = ax.contourf(z_grid, r_grid, values, levels=levels, cmap="viridis")
    fig.colorbar(contour, ax=ax, label=f"{contour_choice} ({units})")

    nr, nz = values.shape
    for i in range(0, nr, max(1, nr // 16)):
        ax.plot(z_grid[i, :], r_grid[i, :], color="white", alpha=0.32, linewidth=0.45)
    for j in range(0, nz, max(1, nz // 20)):
        ax.plot(z_grid[:, j], r_grid[:, j], color="white", alpha=0.32, linewidth=0.45)

    z1_in = bundle["z1_in"]
    z2_in = bundle["z2_in"]
    intake_center = 0.5 * (z1_in + z2_in)
    intake_thickness = z2_in - z1_in

    # Visual cue for the physical intake-port band on the pump body.
    ax.axvspan(z1_in, z2_in, color="white", alpha=0.14)
    ax.axvline(z1_in, color="white", linestyle="--", linewidth=1.2)
    ax.axvline(z2_in, color="white", linestyle="--", linewidth=1.2)
    ax.annotate(
        "Physical fluid intake\n(port band)",
        xy=(intake_center, float(np.nanmax(r_grid)) * 0.98),
        xytext=(intake_center + 0.8 * intake_thickness, float(np.nanmax(r_grid)) * 0.86),
        color="white",
        fontsize=9,
        ha="left",
        va="top",
        arrowprops={"arrowstyle": "->", "color": "white", "lw": 1.1},
    )
    ax.set(
        xlabel="Axial distance from inlet (in)",
        ylabel="Radius from well axis (in)",
        title=f"2-D flow field and finite-difference mesh: {contour_choice}",
    )
    return fig


def plot_bubble_trajectories(bundle):
    fig, ax = plt.subplots(figsize=(9, 5.2), constrained_layout=True)
    z1_in, z2_in = bundle["z1_in"], bundle["z2_in"]
    ax.axhspan(0, bundle["r_p_in"], color="0.75", label="Pump body")
    ax.axvspan(z1_in, z2_in, color="tab:blue", alpha=0.12, label="Physical fluid intake")
    ax.axvline(z1_in, color="tab:blue", linestyle="--", linewidth=1)
    ax.axvline(z2_in, color="tab:blue", linestyle="--", linewidth=1)
    ax.axhline(bundle["r_c_in"], color="black", linewidth=2, label="Casing wall")

    labels = set()
    for r0, _, vented, r_path, z_path in bundle["trajectories"]:
        label = "Vented to annulus (separated)" if vented else "Captured by pump"
        color = "tab:green" if vented else "tab:red"
        ax.plot(z_path, r_path, color=color, linewidth=1.3,
                label=None if label in labels else label)
        ax.plot(z_path[0], r_path[0], "o", color=color, markersize=3)
        labels.add(label)
    ax.set(
        xlabel="Axial distance from inlet (in)",
        ylabel="Radius from well axis (in)",
        title=f"Gas bubble trajectories (E = {bundle['result']['E']:.3f})",
        ylim=(bundle["r_p_in"] * 0.9, bundle["r_c_in"] * 1.03),
    )
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=5)
    ax.grid(alpha=0.25)
    return fig


def plot_fit_diagnostics(bundle, selected_casing, selected_rate):
    rows = pd.DataFrame(bundle["rows"])
    fit = bundle["fit"]
    rows["surrogate_E"] = fit["E_hat"]
    rows["residual"] = rows["E"] - rows["surrogate_E"]

    fig, (ax_fit, ax_residual) = plt.subplots(1, 2, figsize=(11, 4.2))
    colors = {5.0: "tab:blue", 6.0: "tab:orange", 8.0: "tab:green"}
    for casing, group in rows.groupby("casing_id_in"):
        group = group.sort_values("q_l_bpd")
        color = colors[casing]
        ax_fit.plot(
            group["q_l_bpd"],
            group["E"],
            marker="o",
            linestyle="none",
            color=color,
            label=f"{casing:g}-in casing: 2-D",
        )
        ax_fit.plot(
            group["q_l_bpd"],
            group["surrogate_E"],
            linestyle="-",
            color=color,
            alpha=0.85,
            label=f"{casing:g}-in casing: surrogate",
        )
        ax_residual.scatter(
            group["q_l_bpd"],
            group["residual"],
            color=color,
            label=f"{casing:g}-in casing",
        )

    selected_e_2d = bundle["result"]["E"]
    selected_e_surrogate = bundle["surrogate_e"]
    ax_fit.scatter(
        [selected_rate, selected_rate],
        [selected_e_2d, selected_e_surrogate],
        marker="*",
        s=150,
        edgecolor="black",
        linewidth=0.7,
        color=["black", "white"],
        zorder=5,
        label=f"Selected {selected_casing:g}-in case (2-D / surrogate)",
    )
    selected_residual = selected_e_2d - selected_e_surrogate
    ax_residual.scatter(
        [selected_rate],
        [selected_residual],
        marker="*",
        s=150,
        edgecolor="black",
        color="white",
        zorder=5,
        label="Selected case",
    )

    ax_fit.set(xlabel="Liquid rate (bbl/day)", ylabel="Separation efficiency E")
    ax_fit.set_ylim(-0.05, 1.05)
    ax_fit.set_title("Training cases: 2-D model vs surrogate")
    ax_fit.legend(fontsize=8, ncol=2)
    ax_fit.grid(alpha=0.25)
    ax_residual.axhline(0, color="black", linewidth=0.8)
    ax_residual.set(
        xlabel="Liquid rate (bbl/day)",
        ylabel="2-D E − surrogate E",
        title="In-sample residuals",
    )
    ax_residual.legend(fontsize=8)
    ax_residual.grid(alpha=0.25)
    return fig


st.set_page_config(page_title="Marquez Separation Model Comparison", layout="wide")
st.title("Natural Gas Separation Model Comparison")
st.caption("Compare the 2-D trajectory model with a logistic surrogate fitted to its outputs.")

with st.sidebar:
    st.header("Operating conditions")
    st.session_state.setdefault("casing_slider", 6.0)
    st.session_state.setdefault("casing_input", 6.0)

    def _from_slider():
        st.session_state["casing_input"] = st.session_state["casing_slider"]

    def _from_input():
        st.session_state["casing_slider"] = min(
            max(st.session_state["casing_input"], 4.5), 9.0
        )

    st.slider("Casing ID (in)", 4.5, 9.0, step=0.05, key="casing_slider", on_change=_from_slider)
    casing_id = st.number_input(
        "Casing ID, exact value (in)", min_value=4.25, max_value=14.0,
        step=0.01, format="%.2f", key="casing_input", on_change=_from_input,
    )
    liquid_rate = st.slider("Liquid rate (bbl/day)", 300, 3000, 1500, step=100)
    glr = st.slider("Gas-liquid ratio (scf/STB)", 0, 10000, 1000, step=100)
    st.caption("Free gas at intake conditions is computed as q_l × GLR × Bg (dissolved gas is not modeled).")
    pressure_psia = st.number_input("Intake pressure (psia)", 50.0, 5000.0, 300.0, step=25.0)
    temp_f = st.number_input("Intake temperature (°F)", 60.0, 400.0, 100.0, step=5.0)
    gas_sg = st.number_input("Gas gravity (air = 1)", 0.55, 1.2, 0.70, step=0.05)
    z_factor = st.number_input("Gas Z-factor", 0.5, 1.2, 0.90, step=0.01)
    st.header("Numerical resolution")
    nr = st.select_slider("Radial mesh points", options=[15, 21, 31, 41], value=15)
    nz = st.select_slider("Axial mesh points", options=[31, 41, 61, 81], value=31)
    n_bubbles = st.select_slider("Tracked bubbles", options=[11, 15, 21, 25], value=11)
    contour_choice = st.selectbox(
        "Contour field",
        ["Liquid speed magnitude", "Axial liquid velocity", "Radial liquid velocity"],
    )
    st.caption("Pressure and gas-void-fraction contours are unavailable: this model does not calculate those fields.")
    run_clicked = st.button("Run comparison", type="primary", use_container_width=True)

if run_clicked:
    with st.spinner("Running the 2-D cases and fitting the surrogate…"):
        st.session_state["comparison"] = run_case(
            casing_id, liquid_rate, glr, pressure_psia, temp_f, gas_sg, z_factor,
            nr, nz, n_bubbles
        )

bundle = st.session_state.get("comparison")
if bundle is None:
    st.info("Choose conditions and select **Run comparison** to calculate both predictions.")
else:
    result = bundle["result"]
    fit = bundle["fit"]
    difference = result["E"] - bundle["surrogate_e"]
    metric_cols = st.columns(4)
    metric_cols[0].metric("2-D separation efficiency", f'{result["E"]:.3f}')
    metric_cols[1].metric("Surrogate efficiency", f'{bundle["surrogate_e"]:.3f}')
    metric_cols[2].metric("Difference (2-D − surrogate)", f"{difference:+.3f}")
    metric_cols[3].metric("Surrogate training R²", f'{fit["r2"]:.3f}')

    st.subheader("Alhanati and Marquez models")
    pub_cols = st.columns(3)
    for col, (name, value) in zip(pub_cols, bundle["published"].items()):
        col.metric(name, f"{value:.3f}", f"{value - result['E']:+.3f} vs 2-D", delta_color="off")
    st.pyplot(plot_model_sweep(bundle, casing_id, liquid_rate, glr), clear_figure=True)

    st.caption(
        f'Case: {casing_id:g}-in casing, {liquid_rate:,} bbl/day liquid, GLR {glr:,} scf/STB. '
        f'Local gas rate: {bundle["gas_rate_bpd"]:.1f} bbl/day (Bg = {bundle["pvt"]["bg_bbl_scf"]*5.615:.5f} ft³/scf, '
        f'ρg = {bundle["pvt"]["rho_g"]:.2f} lbm/ft³). The 2-D/surrogate fluids keep ρg = 1 lbm/ft³.'
    )
    st.warning(
        "This surrogate is fitted to simulated 2-D outputs, not independent measurements. "
        "Its R² is in-sample and should not be interpreted as experimental validation."
    )
    st.info(
        "In the current 2-D implementation, GLR changes the reported gas-rate split but not "
        "the separation efficiency; the bubble trajectories do not use gas rate."
    )
    if not fit["converged"]:
        st.error("The MLE optimizer did not report convergence. Treat the surrogate comparison with caution.")

    st.subheader("Mesh and selected flow-field contour")
    st.pyplot(plot_mesh_contour(bundle, contour_choice), clear_figure=True)
    st.subheader("Bubble trajectories")
    st.pyplot(plot_bubble_trajectories(bundle), clear_figure=True)
    st.caption(
        "Each line is one bubble released at the inlet (z = 0), spaced by equal annular area. "
        "Green bubbles rise past the top of the intake and are vented; red bubbles reach the pump wall."
    )
    st.subheader("Surrogate fit diagnostics")
    st.pyplot(
        plot_fit_diagnostics(bundle, casing_id, liquid_rate),
        clear_figure=True,
    )
