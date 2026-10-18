# Marquez separation model comparison

This Streamlit app compares the existing 2-D bubble-trajectory model with a
logistic surrogate fitted to a small sweep of that model. It also draws the
finite-difference mesh over a selected liquid-velocity contour.

## Run

From this folder, create/activate a Python environment, install the listed
dependencies, then run:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The first comparison fits the surrogate from 21 unique 2-D cases (three casing
sizes by seven liquid rates). The default resolution is 15 radial points, 31
axial points, and 11 tracked bubbles to keep the first run responsive. Higher
resolutions are available in the sidebar. Subsequent runs with the same
mesh/bubble resolution reuse the cached training sweep.

## Interpretation and limitations

- The surrogate is calibrated to the 2-D model, not to independent experimental
  data. Its displayed R² is an in-sample fit statistic.
- The current 2-D implementation does not calculate pressure or gas void
  fraction (GVF), so the contour choices are liquid speed, axial velocity, and
  radial velocity.
- Gas rate is not used in the current bubble-trajectory calculation. Changing
  GLR (scf/STB, 0–10,000) is converted to local free-gas rate with Bg from the
  intake pressure, temperature, gas gravity and Z. It affects the Marquez
  mechanistic model but not the 2-D model, surrogate, Alhanati or Marquez
  simplified efficiencies. Dissolved gas (Rs) is not modeled.
- The fitted data use the available 5-, 6-, and 8-inch casing cases and the
  selected mesh resolution; interpret predictions outside that range cautiously.
