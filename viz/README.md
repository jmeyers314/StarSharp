# Rubin optical model visualization

An interactive browser view of the Rubin/LSST optical model: drag the 50 AOS
degrees of freedom (or their SVD "v-modes") and watch spot diagrams, corner
donuts, and corner Zernikes respond in real time.

The browser does no ray tracing.  It renders a first-order model,

    spot = spot_xy + sum_i S[i] * dof[i]

from a precomputed sensitivity matrix.  Everything it needs lives in two
generated files, `model.f32` (a flat float32 blob) and `model_meta.json` (which
block lives at which offset).  Both are gitignored — they are ~70 MiB and are
regenerated from StarSharp rather than versioned.

## Generating the data

From the repo root, with the LSST stack and `$TS_OFC_DIR` set up:

```bash
python bootstrap/make_viz_model.py                       # v3.14, r band
python bootstrap/make_viz_model.py --version 1000 --band i
```

This writes `viz/model.f32` and `viz/model_meta.json`.  Run
`make_viz_model.py --help` for the sampling and truncation knobs.  A full run
takes a while; for a quick structural check, coarsen it:

```bash
python bootstrap/make_viz_model.py \
    --science-nrad 2 --pupil-nrad 3 --donut-nrad 3 \
    --dz-field-nrad 3 --zk-rings 4 --outdir /tmp/vizsmoke
```

Note that a coarse `--dz-field-nrad` cannot constrain a high `--kmax`: the
double-Zernike fit needs comfortably more field points than field orders, so
coarse runs are good for checking plumbing and nothing else.

`$TS_OFC_DIR` is required, because one of the two sensitivity matrices is read
from ts_ofc's `policy/`.  The generator fails fast with a clear message if it
is unset.

## Serving

The page is static but fetches two files, so `file://` will not work — the
fetches are blocked as cross-origin.  Serve the directory:

```bash
cd viz && python -m http.server 8000
# then open http://localhost:8000/
```

`index.html` pulls deck.gl from a CDN, so the first load needs network access.

## The two provenance toggles

The v-mode basis comes from an SVD of a double-Zernike sensitivity matrix
scaled by per-DOF normalization weights.  Both inputs have two defensible
sources, and the viz ships both of each so they can be compared without
regenerating anything:

| Toggle | `StarSharp` | `ts_ofc` |
| --- | --- | --- |
| **FWHM wts** | measured by raytracing (`bootstrap/make_fwhm_weights.py`) | derived from ts_ofc's linearized double-Zernike sensitivity |
| **Sens matrix** | raytraced fresh at the chosen version/band | ts_ofc's operational `lsst_sensitivity_dz_31_29_50.yaml` |

The FWHM weights agree to ~12% in the median, with focus-like DOF agreeing to
~0.5%.  The exponents applied to the weights (`range**a * fwhm**b`) are
editable in the UI; `a = 0.5, b = -0.5` is the meaningful default, since
`a - b == 1` is what makes each weight carry its own DOF's physical unit to the
first power.  Other combinations leave a fractional power of the DOF unit and
are not physically interpretable.

The generator applies one convention fix to ts_ofc's matrix: the documented
overall sign flip, same as `default_sensitivity`. That alone reconciles the two
— per-DOF correlation has a median of +0.97, with 26 of 50 DOF above +0.9 and
none below −0.9. The residual disagreement is real physics, not convention:
ts_ofc's matrix is a different optics version fit by a different method.

When comparing the two sources, **compare per-DOF, not in aggregate.** A single
correlation over the whole flattened matrix is dominated by `M2_rx`/`M2_ry`,
whose rows carry 20–30× the norm of everything else, so it tells you about those
two DOF and nothing more. That statistic also makes a spurious 180° field-axis
rotation look like an improvement; per-DOF it is clearly worse.

Both matrices have their `k = 0` rows and `j < 4` columns zeroed before the
SVD: field order 0 is not meaningful under these conventions, and
piston/tip/tilt move the PSF centroid rather than changing its shape.

## Files

- `index.html` — layout, styling, and control markup
- `main.js` — loader, first-order model evaluation, deck.gl rendering, and the
  in-browser SVD (Jacobi eigendecomposition of `A.T A`) that derives v-modes
  live from whichever source is selected
- `model.f32`, `model_meta.json` — generated, gitignored

## History

This started as a standalone repo (`rubin_optical_model_visualization`) whose
generator tracked a much older StarSharp API.  It was folded into StarSharp so
the generator sits beside the `bootstrap/` scripts producing the weights it
consumes, and so API changes surface in one place.  The old repo's vendored
`star_sharp.py` and its detailed per-detector height map were dropped; the viz
never used the height map (`include_chip_heights=False` throughout).
