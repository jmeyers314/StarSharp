let DeckGL;
let OrthographicView;
let ScatterplotLayer;
let COORDINATE_SYSTEM;

let N;
const WORLD_RADIUS = 2;
const VIEW_PADDING = 0.9;
const VALUE_DECIMALS = 3;
const CONTROL_NAMES = [
    'M2 dz [µm]', 'M2 dx [µm]', 'M2 dy [µm]', 'M2 rx [arcsec]', 'M2 ry [arcsec]',
    'Cam dz [µm]', 'Cam dx [µm]', 'Cam dy [µm]', 'Cam rx [arcsec]', 'Cam ry [arcsec]',
    'M1M3 B1 [µm]', 'M1M3 B2 [µm]', 'M1M3 B3 [µm]', 'M1M3 B4 [µm]', 'M1M3 B5 [µm]',
    'M1M3 B6 [µm]', 'M1M3 B7 [µm]', 'M1M3 B8 [µm]', 'M1M3 B9 [µm]', 'M1M3 B10 [µm]',
    'M1M3 B11 [µm]', 'M1M3 B12 [µm]', 'M1M3 B13 [µm]', 'M1M3 B14 [µm]', 'M1M3 B15 [µm]',
    'M1M3 B16 [µm]', 'M1M3 B17 [µm]', 'M1M3 B18 [µm]', 'M1M3 B19 [µm]', 'M1M3 B20 [µm]',
    'M2 B1 [µm]', 'M2 B2 [µm]', 'M2 B3 [µm]', 'M2 B4 [µm]', 'M2 B5 [µm]',
    'M2 B6 [µm]', 'M2 B7 [µm]', 'M2 B8 [µm]', 'M2 B9 [µm]', 'M2 B10 [µm]',
    'M2 B11 [µm]', 'M2 B12 [µm]', 'M2 B13 [µm]', 'M2 B14 [µm]', 'M2 B15 [µm]',
    'M2 B16 [µm]', 'M2 B17 [µm]', 'M2 B18 [µm]', 'M2 B19 [µm]', 'M2 B20 [µm]'
];
const CONTROL_RANGES = [
  200.0, 2500.0, 2500.0, 120.0, 120.0,
  200.0, 6000.0, 6000.0, 120.0, 120.0,
  2.5, 2.5, 2.5, 2.0, 2.0,
  2.0, 2.0, 1.25, 1.25, 1.25,
  1.25, 1.0, 1.0, 1.0, 1.0,
  1.0, 0.5, 0.5, 0.5, 0.5,

  2.5, 2.5, 2.5, 2.5, 2.0,
  2.0, 2.0, 2.0, 2.0, 1.25,
  1.0, 1.0, 1.0, 1.0, 1.0,
  0.5, 0.5, 0.5, 0.5, 0.5,
];

let K = CONTROL_NAMES.length;
const DONUT_CORNERS = 4;
const DONUT_ZERNIKE_TERMS = 29;
const WHEEL_ZOOM_SENSITIVITY = 0.0008;
const WHEEL_ZOOM_MAX_STEP = 0.12;
const MICRONS_PER_PIXEL = 10;   // LSST detector pixel size
const SCALE_BAR_SPOT_PIXELS = 10;   // spot scale bar size in detector pixels
const SCALE_BAR_DONUT_PIXELS = 50;  // donut scale bar size in detector pixels
const SPOT_SCALE = 3e-3;
const DONUT_SPOT_SCALE = 5e-4;
const ARCSEC_PER_DEGREE = 3600;
const ANGULAR_DOF_INDICES = new Set([3, 4, 8, 9]);
const DEFAULT_RANGE_NORM_EXP = 0.5;
const DEFAULT_FWHM_NORM_EXP = -0.5;
const DEFAULT_USE_DOF = '0-16,30-34';
const DEFAULT_NKEEP = 12;
// The two independent provenance choices the viz exposes.  'starsharp' weights
// are raytraced; 'ts_ofc' are the operational values derived from ts_ofc's
// linearized double-Zernike sensitivity.
const WEIGHT_SOURCES = ['starsharp', 'ts_ofc'];
const SOURCE_LABELS = {starsharp: 'StarSharp', ts_ofc: 'ts_ofc'};
const DEFAULT_FWHM_SOURCE = 'starsharp';
const DEFAULT_SENS_SOURCE = 'starsharp';
const DEFAULT_DZ_FIELD_MAX = 3;
const DEFAULT_DZ_PUPIL_MAX = 11;

let p = new Float32Array(K);
let scienceN = 0;
let donutN = 0;
let scienceX0, scienceY0, scienceSx, scienceSy;
let scienceFieldX, scienceFieldY;
let donutX0, donutY0, donutSx, donutSy;
let donutFieldX, donutFieldY;
let donutCenterX, donutCenterY;
let spotScale = SPOT_SCALE;
let donutSpotScale = DONUT_SPOT_SCALE;
let donutClockAngleDeg = 10;
let donutClockRadiusScale = 1.4;
let zk0 = new Float32Array(0);
let dZk = new Float32Array(0);
let modelRadius = WORLD_RADIUS;
let positions = new Float32Array(0);
let data = [];

// Vmode state
let norm = null;          // Float64Array(K) — normalization weights
let rangeWeights = null;  // Float64Array(K)
let fwhmWeights = null;   // Float64Array(K) — the currently selected source
let fwhmWeightsBySource = {};  // source name -> Float32Array(K)
let wfSensBySource = {};       // source name -> Float32Array(rows × K)
let fwhmSource = DEFAULT_FWHM_SOURCE;
let sensSource = DEFAULT_SENS_SOURCE;
let rangeNormExp = DEFAULT_RANGE_NORM_EXP;
let fwhmNormExp = DEFAULT_FWHM_NORM_EXP;
let wfSens = null;        // Float32Array — wavefront sensitivity matrix (nObs × K)
let wfSensRows = 0;       // number of observation rows in wfSens
let wfSensNFieldDz = 0;   // number of field DZ terms in wfSens rows
let wfSensNPupilZk = 0;   // number of pupil Zernike terms in each row of wfSens
let currentUseDof = [];   // Int array of active DOF indices
let currentNkeep = DEFAULT_NKEEP;
let Vh = null;            // Float64Array — (nkeep × nActive) mixing matrix
let fullMixMatrix = null; // Float64Array — (nActive × nActive) full eigenvector matrix
let singularValues = null; // Float64Array — singular values for all active v-modes
let nActive = 0;          // len(currentUseDof)
let vValues = [];         // current vmode slider values
let vmodeSliderInputs = [];
let vmodeParamInputs = [];
const loadTelemetry = {
  startupStartIso: null,
  startupEndIso: null,
  startupDurationMs: null,
  metaFetchDurationMs: null,
  metaBytes: null,
  metaHttpStatus: null,
  modelFetchDurationMs: null,
  modelBytesExpected: null,
  modelBytesReceived: null,
  modelHttpStatus: null,
  modelUsedStreamReader: null,
  modelDownloadCompleted: false,
};

function nowMs() {
  if (typeof performance !== 'undefined' && performance && typeof performance.now === 'function') {
    return performance.now();
  }
  return Date.now();
}

function formatDuration(value) {
  if (!Number.isFinite(value) || value < 0) return 'unknown';
  return value.toFixed(1);
}

function estimateRadius(x, y) {
  let maxR2 = 0;
  for (let i = 0; i < x.length; i++) {
    const r2 = x[i] * x[i] + y[i] * y[i];
    if (r2 > maxR2) maxR2 = r2;
  }
  return Math.max(1e-6, Math.sqrt(maxR2));
}

function getControlSpec(index) {
  const label = CONTROL_NAMES[index] != null ? CONTROL_NAMES[index] : `p${index}`;
  const range = CONTROL_RANGES[index] != null ? CONTROL_RANGES[index] : 1;
  const step = Math.max(range / 200, 1e-4);
  return {label, range, step};
}

function clampControlValue(index, value) {
  const range = CONTROL_RANGES[index] != null ? CONTROL_RANGES[index] : Infinity;
  return Math.max(-range, Math.min(range, value));
}

function formatControlValue(value) {
  return value.toFixed(VALUE_DECIMALS);
}

function hideCanvas(canvasEl) {
  canvasEl.width = 0;
  canvasEl.height = 0;
  canvasEl.style.width = '0';
  canvasEl.style.height = '0';
}

function setupHiDpiCanvas(canvasEl, widthCssPx, heightCssPx) {
  const dpr = window.devicePixelRatio || 1;
  canvasEl.width = Math.ceil(widthCssPx * dpr);
  canvasEl.height = Math.ceil(heightCssPx * dpr);
  canvasEl.style.width = widthCssPx + 'px';
  canvasEl.style.height = heightCssPx + 'px';

  const ctx = canvasEl.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, widthCssPx, heightCssPx);
  return ctx;
}

function drawUnusedModesOverlay(ctx, plotLeft, plotTop, nModes, currentKeep, cellSize, plotHeight) {
  if (currentKeep >= nModes) return;
  ctx.fillStyle = 'rgba(120, 120, 120, 0.45)';
  ctx.fillRect(plotLeft + currentKeep * cellSize, plotTop, (nModes - currentKeep) * cellSize, plotHeight);
}

function bindEnterEscapeInput(inputEl, onEnter) {
  if (!inputEl) return;
  inputEl.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      onEnter();
    }
    if (e.key === 'Escape') {
      e.preventDefault();
      inputEl.blur();
    }
  });
}

function dofValueToModelUnits(dofIndex, value) {
  if (ANGULAR_DOF_INDICES.has(dofIndex)) {
    return value / ARCSEC_PER_DEGREE;
  }
  return value;
}

function computeClockedCenter(x, y, isIntra) {
  const theta = (isIntra ? donutClockAngleDeg : -donutClockAngleDeg) * Math.PI / 180;
  const c = Math.cos(theta);
  const s = Math.sin(theta);
  return {
    x: donutClockRadiusScale * (c * x - s * y),
    y: donutClockRadiusScale * (s * x + c * y),
  };
}

// --- use_dof string parser (mirrors Python StateFactory) ---
function parseUseDof(s) {
  if (typeof s !== 'string') return [];
  s = s.replace(/\s/g, '').trim();
  if (!s) return [];
  const result = [];
  for (const part of s.split(',')) {
    if (part.includes('-')) {
      const [a, b] = part.split('-').map(Number);
      if (!Number.isInteger(a) || !Number.isInteger(b)) continue;
      for (let i = a; i <= b; i++) result.push(i);
    } else {
      const v = Number(part);
      if (Number.isInteger(v)) result.push(v);
    }
  }
  return [...new Set(result)].sort((a, b) => a - b);
}

// --- Format sorted int array back to compact range string ---
function formatUseDof(arr) {
  if (!arr || arr.length === 0) return '';
  const parts = [];
  let start = arr[0], end = arr[0];
  for (let i = 1; i < arr.length; i++) {
    if (arr[i] === end + 1) {
      end = arr[i];
    } else {
      parts.push(start === end ? String(start) : `${start}-${end}`);
      start = arr[i];
      end = arr[i];
    }
  }
  parts.push(start === end ? String(start) : `${start}-${end}`);
  return parts.join(',');
}

// --- Jacobi eigendecomposition for real symmetric matrix ---
function jacobiEigen(G, maxIter) {
  const n = Math.round(Math.sqrt(G.length));
  if (n * n !== G.length) throw new Error('jacobiEigen: not a square matrix');
  if (typeof maxIter !== 'number') maxIter = 100 * n * n;

  // Copy G into A (we'll destroy it)
  const A = new Float64Array(G);
  // V starts as identity
  const V = new Float64Array(n * n);
  for (let i = 0; i < n; i++) V[i * n + i] = 1;

  for (let iter = 0; iter < maxIter; iter++) {
    // Find largest off-diagonal element
    let maxVal = 0, ip = 0, iq = 0;
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        const v = Math.abs(A[i * n + j]);
        if (v > maxVal) { maxVal = v; ip = i; iq = j; }
      }
    }
    if (maxVal < 1e-15) break;

    const aij = A[ip * n + iq];
    const aii = A[ip * n + ip];
    const ajj = A[iq * n + iq];
    const theta = 0.5 * Math.atan2(2 * aij, aii - ajj);
    const c = Math.cos(theta);
    const s = Math.sin(theta);

    // Rotate rows/cols ip, iq of A
    for (let k = 0; k < n; k++) {
      const aik = A[ip * n + k];
      const ajk = A[iq * n + k];
      A[ip * n + k] = c * aik + s * ajk;
      A[iq * n + k] = -s * aik + c * ajk;
    }
    for (let k = 0; k < n; k++) {
      const aki = A[k * n + ip];
      const akj = A[k * n + iq];
      A[k * n + ip] = c * aki + s * akj;
      A[k * n + iq] = -s * aki + c * akj;
    }

    // Accumulate V
    for (let k = 0; k < n; k++) {
      const vki = V[k * n + ip];
      const vkj = V[k * n + iq];
      V[k * n + ip] = c * vki + s * vkj;
      V[k * n + iq] = -s * vki + c * vkj;
    }
  }

  // Extract eigenvalues from diagonal of A
  const eigenvalues = new Float64Array(n);
  for (let i = 0; i < n; i++) eigenvalues[i] = A[i * n + i];

  // Sort by decreasing eigenvalue, return eigenvectors as columns of V
  const order = Array.from({length: n}, (_, i) => i);
  order.sort((a, b) => eigenvalues[b] - eigenvalues[a]);

  const sortedVals = new Float64Array(n);
  const sortedVecs = new Float64Array(n * n); // each column is an eigenvector
  for (let rank = 0; rank < n; rank++) {
    const orig = order[rank];
    sortedVals[rank] = eigenvalues[orig];
    for (let row = 0; row < n; row++) {
      sortedVecs[row * n + rank] = V[row * n + orig];
    }
  }

  return {values: sortedVals, vectors: sortedVecs};
}

// --- Compute Vh from wavefront sensitivity, norm, useDof, nkeep ---
// Mirrors StateFactory.__init__() from state.py
function computeVh(sens, sensRows, sensK, normArr, useDof, nkeep) {
  const nAct = useDof.length;
  if (nAct === 0 || nkeep <= 0) return null;

  // A_norm = sens * diag(norm)  (column scaling)
  // A_sliced = A_norm[:, useDof]
  // G = A_sliced^T @ A_sliced   (nAct x nAct)
  const G = new Float64Array(nAct * nAct);
  for (let i = 0; i < nAct; i++) {
    for (let j = i; j < nAct; j++) {
      let dot = 0;
      const di = useDof[i];
      const dj = useDof[j];
      const ni = normArr[di];
      const nj = normArr[dj];
      for (let r = 0; r < sensRows; r++) {
        dot += (sens[r * sensK + di] * ni) * (sens[r * sensK + dj] * nj);
      }
      G[i * nAct + j] = dot;
      G[j * nAct + i] = dot;
    }
  }

  const {values, vectors} = jacobiEigen(G);

  // Vh = top nkeep eigenvectors as rows, denormalized
  // vectors is column-major: vectors[row * nAct + col] where col = eigenvector index
  const nk = Math.min(nkeep, nAct);
  const vh = new Float64Array(nk * nAct);
  const fullMatrix = new Float64Array(nAct * nAct);
  const sigma = new Float64Array(nAct);
  for (let mode = 0; mode < nAct; mode++) {
    sigma[mode] = Math.sqrt(Math.max(0, values[mode]));
    for (let j = 0; j < nAct; j++) {
      const val = vectors[j * nAct + mode] * normArr[useDof[j]];
      fullMatrix[mode * nAct + j] = val;
      if (mode < nk) vh[mode * nAct + j] = val;
    }
  }
  return {vh, fullMatrix, singularValues: sigma};
}

async function loadPackedModel(metaUrl, modelUrl) {
  const metaStart = nowMs();
  const metaResponse = await fetch(metaUrl);
  loadTelemetry.metaHttpStatus = metaResponse.status;
  const metaEnd = nowMs();
  loadTelemetry.metaFetchDurationMs = Math.max(0, metaEnd - metaStart);
  const metaLengthHeader = metaResponse.headers.get('content-length');
  const metaBytes = metaLengthHeader ? Number(metaLengthHeader) : NaN;
  loadTelemetry.metaBytes = Number.isFinite(metaBytes) && metaBytes >= 0 ? metaBytes : null;
  if (!metaResponse.ok) {
    throw new Error(`Failed to load ${metaUrl}: ${metaResponse.status} ${metaResponse.statusText}`);
  }
  const meta = await metaResponse.json();

  const modelBuffer = await fetchArrayBufferWithProgress(modelUrl, updateLoadingProgress);
  const packed = new Float32Array(modelBuffer);

  const n = Number(meta.N);
  const k = Number(meta.K);
  if (!Number.isInteger(n) || n <= 0 || !Number.isInteger(k) || k <= 0) {
    throw new Error('Metadata must contain positive integer N and K');
  }

  const layout = meta && meta.layout ? meta.layout : null;
  const spotSpec = layout ? layout.spot_xy : undefined;
  const sxSpec = layout ? layout.Sx : undefined;
  const sySpec = layout ? layout.Sy : undefined;
  const fieldSpec = layout ? layout.field_xy : undefined;
  const donutSpotSpec = layout ? layout.donut_spot_xy : undefined;
  const donutSxSpec = layout ? layout.donut_Sx : undefined;
  const donutSySpec = layout ? layout.donut_Sy : undefined;
  const donutFieldSpec = layout ? layout.donut_field_xy : undefined;
  const rangeWeightsSpec = layout ? layout.range_weights : undefined;
  // Layout v8 split the single fwhm_weights / wf_sens blocks into one per
  // source so the browser can switch between them.  Fall back to the v7 names
  // so older generated models still load.
  const fwhmSpecs = layout ? {
    starsharp: layout.fwhm_starsharp_weights || layout.fwhm_weights,
    ts_ofc: layout.fwhm_ts_ofc_weights,
  } : {};
  const wfSensSpecs = layout ? {
    starsharp: layout.wf_sens_starsharp || layout.wf_sens,
    ts_ofc: layout.wf_sens_ts_ofc,
  } : {};
  const fwhmWeightsSpec = fwhmSpecs.starsharp;
  const wfSensSpec = wfSensSpecs.starsharp;
  if (!spotSpec || !sxSpec || !sySpec || !fieldSpec) {
    throw new Error('Metadata layout must include spot_xy, Sx, Sy, and field_xy blocks');
  }

  const spotOffset = Number(spotSpec.offset_f32);
  const spotLength = Number(spotSpec.length_f32);
  const sxOffset = Number(sxSpec.offset_f32);
  const sxLength = Number(sxSpec.length_f32);
  const syOffset = Number(sySpec.offset_f32);
  const syLength = Number(sySpec.length_f32);
  const fieldOffset = Number(fieldSpec.offset_f32);
  const fieldLength = Number(fieldSpec.length_f32);

  const donutCount = Number(meta.donut_N || 0);
  const hasDonuts = donutCount > 0 && donutSpotSpec && donutSxSpec && donutSySpec && donutFieldSpec;

  let donutSpotOffset = 0;
  let donutSpotLength = 0;
  let donutSxOffset = 0;
  let donutSxLength = 0;
  let donutSyOffset = 0;
  let donutSyLength = 0;
  let donutFieldOffset = 0;
  let donutFieldLength = 0;
  if (hasDonuts) {
    donutSpotOffset = Number(donutSpotSpec.offset_f32);
    donutSpotLength = Number(donutSpotSpec.length_f32);
    donutSxOffset = Number(donutSxSpec.offset_f32);
    donutSxLength = Number(donutSxSpec.length_f32);
    donutSyOffset = Number(donutSySpec.offset_f32);
    donutSyLength = Number(donutSySpec.length_f32);
    donutFieldOffset = Number(donutFieldSpec.offset_f32);
    donutFieldLength = Number(donutFieldSpec.length_f32);
  }

  const hasVmodeData = !!(rangeWeightsSpec && fwhmWeightsSpec && wfSensSpec);
  let rangeWeightsOffset = 0;
  let rangeWeightsLength = 0;
  let fwhmWeightsOffset = 0;
  let fwhmWeightsLength = 0;
  let wfSensOffset = 0;
  let wfSensLength = 0;
  let wfRows = 0;
  let wfCols = 0;
  let wfFieldDz = 0;
  let wfPupilZk = 0;
  if (hasVmodeData) {
    rangeWeightsOffset = Number(rangeWeightsSpec.offset_f32);
    rangeWeightsLength = Number(rangeWeightsSpec.length_f32);
    fwhmWeightsOffset = Number(fwhmWeightsSpec.offset_f32);
    fwhmWeightsLength = Number(fwhmWeightsSpec.length_f32);
    wfSensOffset = Number(wfSensSpec.offset_f32);
    wfSensLength = Number(wfSensSpec.length_f32);
    wfRows = wfSensSpec.shape ? Number(wfSensSpec.shape[0]) : 0;
    wfCols = wfSensSpec.shape ? Number(wfSensSpec.shape[1]) : 0;
    wfFieldDz = Number(wfSensSpec.n_field_dz || 0);
    wfPupilZk = Number(wfSensSpec.n_pupil_zk || 0);
  }

  if (
    ![spotOffset, spotLength, sxOffset, sxLength, syOffset, syLength, fieldOffset, fieldLength].every(Number.isInteger)
  ) {
    throw new Error('All metadata offsets/lengths must be integers');
  }
  if (hasDonuts && ![
    donutSpotOffset,
    donutSpotLength,
    donutSxOffset,
    donutSxLength,
    donutSyOffset,
    donutSyLength,
    donutFieldOffset,
    donutFieldLength,
  ].every(Number.isInteger)) {
    throw new Error('All donut metadata offsets/lengths must be integers');
  }
  if (hasVmodeData && ![
    rangeWeightsOffset,
    rangeWeightsLength,
    fwhmWeightsOffset,
    fwhmWeightsLength,
    wfSensOffset,
    wfSensLength,
  ].every(Number.isInteger)) {
    throw new Error('All vmode metadata offsets/lengths must be integers');
  }

  if (spotLength !== 2 * n || sxLength !== k * n || syLength !== k * n || fieldLength !== 2 * n) {
    throw new Error('Metadata block lengths do not match N/K');
  }
  if (hasDonuts) {
    if (donutSpotLength !== 2 * donutCount || donutSxLength !== k * donutCount || donutSyLength !== k * donutCount || donutFieldLength !== 2 * donutCount) {
      throw new Error('Donut metadata block lengths do not match donut_N/K');
    }
  }
  if (hasVmodeData) {
    if (rangeWeightsLength !== k || fwhmWeightsLength !== k) {
      throw new Error('Vmode weight lengths do not match K');
    }
    if (!Number.isInteger(wfRows) || !Number.isInteger(wfCols) || wfRows <= 0 || wfCols !== k) {
      throw new Error('wf_sens shape is invalid or does not match K');
    }
    if (wfSensLength !== wfRows * wfCols) {
      throw new Error('wf_sens metadata length does not match its shape');
    }
    if (wfFieldDz > 0 && wfPupilZk > 0 && wfRows !== wfFieldDz * wfPupilZk) {
      throw new Error('wf_sens n_field_dz and n_pupil_zk do not match wf_sens rows');
    }
  }

  const zk0Spec = layout ? layout.zk0 : undefined;
  const dzkSpec = layout ? layout.dzk : undefined;
  const hasZernikes = !!(zk0Spec && dzkSpec);
  let zk0Offset = 0, zk0Length = 0, dzkOffset = 0, dzkLength = 0;
  let zkCorners = 0, zkTerms = 0;
  if (hasZernikes) {
    zk0Offset = Number(zk0Spec.offset_f32);
    zk0Length = Number(zk0Spec.length_f32);
    dzkOffset = Number(dzkSpec.offset_f32);
    dzkLength = Number(dzkSpec.length_f32);
    zkCorners = zk0Spec.shape ? Number(zk0Spec.shape[0]) : DONUT_CORNERS;
    zkTerms = zk0Spec.shape ? Number(zk0Spec.shape[1]) : DONUT_ZERNIKE_TERMS;
  }

  // Every per-source block counts toward the required length, not just the
  // primary one, or a truncated file would slice garbage for the alternates.
  const sourceBlockEnds = [];
  for (const specs of [fwhmSpecs, wfSensSpecs]) {
    for (const src of WEIGHT_SOURCES) {
      const spec = specs[src];
      if (!spec) continue;
      const end = Number(spec.offset_f32) + Number(spec.length_f32);
      if (Number.isInteger(end)) sourceBlockEnds.push(end);
    }
  }

  const totalExpected = Math.max(
    spotOffset + spotLength,
    sxOffset + sxLength,
    syOffset + syLength,
    fieldOffset + fieldLength,
    hasDonuts ? donutSpotOffset + donutSpotLength : 0,
    hasDonuts ? donutSxOffset + donutSxLength : 0,
    hasDonuts ? donutSyOffset + donutSyLength : 0,
    hasDonuts ? donutFieldOffset + donutFieldLength : 0,
    hasVmodeData ? rangeWeightsOffset + rangeWeightsLength : 0,
    hasZernikes ? zk0Offset + zk0Length : 0,
    hasZernikes ? dzkOffset + dzkLength : 0,
    ...sourceBlockEnds
  );
  if (packed.length < totalExpected) {
    throw new Error(`Packed model too short: expected at least ${totalExpected} float32 values, got ${packed.length}`);
  }

  const spotFlat = packed.subarray(spotOffset, spotOffset + spotLength);
  const fieldFlat = packed.subarray(fieldOffset, fieldOffset + fieldLength);
  const x = new Float32Array(n);
  const y = new Float32Array(n);
  const loadedFieldX = new Float32Array(n);
  const loadedFieldY = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    x[i] = spotFlat[2 * i];
    y[i] = spotFlat[2 * i + 1];
    loadedFieldX[i] = fieldFlat[2 * i];
    loadedFieldY[i] = fieldFlat[2 * i + 1];
  }

  const loadedSx = new Float32Array(packed.subarray(sxOffset, sxOffset + sxLength));
  const loadedSy = new Float32Array(packed.subarray(syOffset, syOffset + syLength));

  let loadedDonutX = new Float32Array(0);
  let loadedDonutY = new Float32Array(0);
  let loadedDonutFieldX = new Float32Array(0);
  let loadedDonutFieldY = new Float32Array(0);
  let loadedDonutSx = new Float32Array(0);
  let loadedDonutSy = new Float32Array(0);
  if (hasDonuts) {
    const donutSpotFlat = packed.subarray(donutSpotOffset, donutSpotOffset + donutSpotLength);
    const donutFieldFlat = packed.subarray(donutFieldOffset, donutFieldOffset + donutFieldLength);
    loadedDonutX = new Float32Array(donutCount);
    loadedDonutY = new Float32Array(donutCount);
    loadedDonutFieldX = new Float32Array(donutCount);
    loadedDonutFieldY = new Float32Array(donutCount);
    for (let i = 0; i < donutCount; i++) {
      loadedDonutX[i] = donutSpotFlat[2 * i];
      loadedDonutY[i] = donutSpotFlat[2 * i + 1];
      loadedDonutFieldX[i] = donutFieldFlat[2 * i];
      loadedDonutFieldY[i] = donutFieldFlat[2 * i + 1];
    }
    loadedDonutSx = new Float32Array(packed.subarray(donutSxOffset, donutSxOffset + donutSxLength));
    loadedDonutSy = new Float32Array(packed.subarray(donutSyOffset, donutSyOffset + donutSyLength));
  }

  let loadedZk0 = new Float32Array(0);
  let loadedDzk = new Float32Array(0);
  if (hasZernikes) {
    loadedZk0 = new Float32Array(packed.subarray(zk0Offset, zk0Offset + zk0Length));
    loadedDzk = new Float32Array(packed.subarray(dzkOffset, dzkOffset + dzkLength));
  }

  let loadedRangeWeights = new Float32Array(0);
  const loadedFwhmWeights = {};
  const loadedWfSens = {};
  if (hasVmodeData) {
    loadedRangeWeights = new Float32Array(packed.subarray(rangeWeightsOffset, rangeWeightsOffset + rangeWeightsLength));

    // Zero out rows that are meaningless for optical-mode decomposition:
    //   k = 0  — field DZ order 0 is undefined/meaningless with our conventions
    //   j < 4  — piston/tip/tilt shift the PSF centroid, not its shape
    const maskRows = (arr) => {
      if (wfPupilZk <= 0) return arr;
      for (let r = 0; r < wfRows; r++) {
        const k = Math.floor(r / wfPupilZk);  // field DZ order
        const z = r % wfPupilZk;              // pupil Zernike index
        if (k === 0 || z < 4) {
          const rowStart = r * wfCols;
          arr.fill(0, rowStart, rowStart + wfCols);
        }
      }
      return arr;
    };

    for (const src of WEIGHT_SOURCES) {
      const fwhmSpec = fwhmSpecs[src];
      if (fwhmSpec) {
        const off = Number(fwhmSpec.offset_f32);
        const len = Number(fwhmSpec.length_f32);
        if (len === k) {
          loadedFwhmWeights[src] = new Float32Array(packed.subarray(off, off + len));
        }
      }
      const sensSpec = wfSensSpecs[src];
      if (sensSpec) {
        const off = Number(sensSpec.offset_f32);
        const len = Number(sensSpec.length_f32);
        const rows = sensSpec.shape ? Number(sensSpec.shape[0]) : 0;
        const cols = sensSpec.shape ? Number(sensSpec.shape[1]) : 0;
        // Every source must share the primary block's shape, since the browser
        // swaps them into the same SVD without re-deriving dimensions.
        if (rows === wfRows && cols === wfCols && len === rows * cols) {
          loadedWfSens[src] = maskRows(
            new Float32Array(packed.subarray(off, off + len))
          );
        }
      }
    }
  }

  return {
    x,
    y,
    fieldX: loadedFieldX,
    fieldY: loadedFieldY,
    sx: loadedSx,
    sy: loadedSy,
    donutX: loadedDonutX,
    donutY: loadedDonutY,
    donutFieldX: loadedDonutFieldX,
    donutFieldY: loadedDonutFieldY,
    donutSx: loadedDonutSx,
    donutSy: loadedDonutSy,
    donutN: donutCount,
    donutNfieldIntra: Number(meta.donut_nfield_intra || 0),
    donutNfieldExtra: Number(meta.donut_nfield_extra || 0),
    donutNray: Number(meta.donut_nray || 0),
    donutClockAngleDeg: Number(meta.donut_clock_angle_deg || 10),
    donutClockRadiusScale: Number(meta.donut_clock_radius_scale || 1.4),
    rangeWeights: loadedRangeWeights,
    fwhmWeightsBySource: loadedFwhmWeights,
    wfSensBySource: loadedWfSens,
    wfSensRows: wfRows,
    wfSensNFieldDz: wfFieldDz,
    wfSensNPupilZk: wfPupilZk,
    zk0: loadedZk0,
    dzk: loadedDzk,
    zkCorners,
    zkTerms,
    k,
  };
}

async function fetchArrayBufferWithProgress(url, onProgress) {
  const fetchStart = nowMs();
  const response = await fetch(url);
  loadTelemetry.modelHttpStatus = response.status;
  if (!response.ok) {
    throw new Error(`Failed to load ${url}: ${response.status} ${response.statusText}`);
  }

  const totalHeader = response.headers.get('content-length');
  const totalBytes = totalHeader ? Number(totalHeader) : NaN;
  loadTelemetry.modelBytesExpected = Number.isFinite(totalBytes) && totalBytes >= 0 ? totalBytes : null;
  const responseBody = response.body;
  const reader = responseBody && typeof responseBody.getReader === 'function'
    ? responseBody.getReader()
    : null;
  loadTelemetry.modelUsedStreamReader = !!reader;

  if (!reader) {
    const fallback = await response.arrayBuffer();
    loadTelemetry.modelBytesReceived = fallback.byteLength;
    loadTelemetry.modelDownloadCompleted = true;
    loadTelemetry.modelFetchDurationMs = Math.max(0, nowMs() - fetchStart);
    if (typeof onProgress === 'function') {
      onProgress(fallback.byteLength, fallback.byteLength, true);
    }
    return fallback;
  }

  let receivedBytes = 0;
  const chunks = [];
  while (true) {
    const {done, value} = await reader.read();
    if (done) break;
    if (value) {
      chunks.push(value);
      receivedBytes += value.byteLength;
      loadTelemetry.modelBytesReceived = receivedBytes;
      if (typeof onProgress === 'function') {
        onProgress(receivedBytes, totalBytes, false);
      }
    }
  }

  const merged = new Uint8Array(receivedBytes);
  let offset = 0;
  for (const chunk of chunks) {
    merged.set(chunk, offset);
    offset += chunk.byteLength;
  }

  if (typeof onProgress === 'function') {
    onProgress(receivedBytes, totalBytes, true);
  }
  loadTelemetry.modelBytesReceived = receivedBytes;
  loadTelemetry.modelDownloadCompleted = true;
  loadTelemetry.modelFetchDurationMs = Math.max(0, nowMs() - fetchStart);
  return merged.buffer;
}

function initModel(model) {
  const initialX = model.x;
  const initialY = model.y;
  const initialFieldX = model.fieldX;
  const initialFieldY = model.fieldY;
  const initialSx = model.sx;
  const initialSy = model.sy;
  const loadedK = model.k;

  scienceN = initialX.length;
  donutN = model.donutN || 0;
  N = scienceN + donutN;
  if (Number.isInteger(loadedK) && loadedK > 0) {
    K = loadedK;
  }
  p = new Float32Array(K);
  scienceX0 = initialX;
  scienceY0 = initialY;
  scienceFieldX = initialFieldX;
  scienceFieldY = initialFieldY;
  scienceSx = new Float32Array(K * scienceN);
  scienceSy = new Float32Array(K * scienceN);

  donutX0 = model.donutX || new Float32Array(0);
  donutY0 = model.donutY || new Float32Array(0);
  donutFieldX = model.donutFieldX || new Float32Array(0);
  donutFieldY = model.donutFieldY || new Float32Array(0);
  donutSx = new Float32Array(K * donutN);
  donutSy = new Float32Array(K * donutN);
  donutCenterX = new Float32Array(donutN);
  donutCenterY = new Float32Array(donutN);

  spotScale = SPOT_SCALE;
  donutSpotScale = DONUT_SPOT_SCALE;
  donutClockAngleDeg = model.donutClockAngleDeg || 10;
  donutClockRadiusScale = model.donutClockRadiusScale || 1.4;

  positions = new Float32Array(N * 2);
  data = Array.from({length: N}, (_, i) => i);

  if (!scienceFieldX || !scienceFieldY || scienceFieldX.length !== scienceN || scienceFieldY.length !== scienceN) {
    throw new Error('Missing or invalid field_xy arrays in loaded model');
  }

  if (!initialSx || !initialSy) {
    throw new Error('Missing Sx/Sy arrays in loaded model');
  }

  if (initialSx.length !== K * scienceN || initialSy.length !== K * scienceN) {
    throw new Error('Loaded sensitivity sizes do not match K*N');
  }

  if (donutN > 0) {
    if (donutX0.length !== donutN || donutY0.length !== donutN || donutFieldX.length !== donutN || donutFieldY.length !== donutN) {
      throw new Error('Loaded donut arrays do not match donut_N');
    }
    if (!model.donutSx || !model.donutSy || model.donutSx.length !== K * donutN || model.donutSy.length !== K * donutN) {
      throw new Error('Loaded donut sensitivities do not match K*donut_N');
    }
  }

  scienceSx.set(initialSx);
  scienceSy.set(initialSy);
  if (donutN > 0) {
    donutSx.set(model.donutSx);
    donutSy.set(model.donutSy);
  }

  if (model.zk0 && model.zk0.length > 0) {
    zk0 = model.zk0;
    dZk = model.dzk;
  }

  rangeWeights = null;
  fwhmWeights = null;
  fwhmWeightsBySource = {};
  wfSensBySource = {};
  wfSens = null;
  wfSensRows = 0;
  wfSensNFieldDz = 0;
  wfSensNPupilZk = 0;
  const fwhmLoaded = model.fwhmWeightsBySource || {};
  const sensLoaded = model.wfSensBySource || {};
  if (model.rangeWeights && model.rangeWeights.length === K
      && fwhmLoaded[DEFAULT_FWHM_SOURCE] && sensLoaded[DEFAULT_SENS_SOURCE]) {
    rangeWeights = new Float64Array(model.rangeWeights);
    fwhmWeightsBySource = fwhmLoaded;
    wfSensBySource = sensLoaded;
    wfSensRows = Number(model.wfSensRows) || 0;
    wfSensNFieldDz = Number(model.wfSensNFieldDz) || 0;
    wfSensNPupilZk = Number(model.wfSensNPupilZk) || 0;
    // A model may ship only one source (e.g. a v7 file); fall back rather than
    // leaving the selection pointing at nothing.
    if (!fwhmWeightsBySource[fwhmSource]) fwhmSource = DEFAULT_FWHM_SOURCE;
    if (!wfSensBySource[sensSource]) sensSource = DEFAULT_SENS_SOURCE;
    fwhmWeights = new Float64Array(fwhmWeightsBySource[fwhmSource]);
    wfSens = wfSensBySource[sensSource];
  }

  if (donutN > 0) {
    const nfieldIntra = Number(model.donutNfieldIntra || 0);
    const nray = Number(model.donutNray || 0);
    const intraPointCount = nfieldIntra * nray;
    for (let i = 0; i < donutN; i++) {
      const isIntra = i < intraPointCount;
      const center = computeClockedCenter(donutFieldX[i], donutFieldY[i], isIntra);
      donutCenterX[i] = center.x;
      donutCenterY[i] = center.y;
    }
  }

  buildZernikeOutputs();

  let maxR2 = 0;
  for (let i = 0; i < scienceN; i++) {
    const xx = scienceFieldX[i] + scienceX0[i] * spotScale;
    const yy = scienceFieldY[i] + scienceY0[i] * spotScale;
    const r2 = xx * xx + yy * yy;
    if (r2 > maxR2) maxR2 = r2;
  }
  for (let i = 0; i < donutN; i++) {
    const xx = donutCenterX[i] + donutX0[i] * donutSpotScale;
    const yy = donutCenterY[i] + donutY0[i] * donutSpotScale;
    const r2 = xx * xx + yy * yy;
    if (r2 > maxR2) maxR2 = r2;
  }
  modelRadius = Math.max(1e-6, Math.sqrt(maxR2));
}

function updatePositions() {
  for (let i = 0; i < scienceN; i++) {
    positions[2*i] = scienceFieldX[i] + scienceX0[i] * spotScale;
    positions[2*i+1] = scienceFieldY[i] + scienceY0[i] * spotScale;
  }
  for (let i = 0; i < donutN; i++) {
    const idx = scienceN + i;
    positions[2*idx] = donutCenterX[i] + donutX0[i] * donutSpotScale;
    positions[2*idx+1] = donutCenterY[i] + donutY0[i] * donutSpotScale;
  }

  for (let k = 0; k < K; k++) {
    const pk = dofValueToModelUnits(k, p[k]);
    if (pk === 0) continue;
    const scienceOff = k * scienceN;
    for (let i = 0; i < scienceN; i++) {
      positions[2*i] += scienceSx[scienceOff + i] * pk * spotScale;
      positions[2*i+1] += scienceSy[scienceOff + i] * pk * spotScale;
    }

    const donutOff = k * donutN;
    for (let i = 0; i < donutN; i++) {
      const idx = scienceN + i;
      positions[2*idx] += donutSx[donutOff + i] * pk * donutSpotScale;
      positions[2*idx+1] += donutSy[donutOff + i] * pk * donutSpotScale;
    }
  }
}

const vis = document.getElementById('vis');
const gridCanvas = document.getElementById('grid-canvas');
const canvas = document.getElementById('deck-canvas');
const resetControlsBtn = document.getElementById('reset-controls');
const includeIntrinsicsCheckbox = document.getElementById('include-intrinsics');
const mouseCoordsEl = document.getElementById('mouse-coords');
const scaleBarLineEl = document.getElementById('scale-bar-line');
const scaleBarSpotLineEl = document.getElementById('scale-bar-spot-line');
const scaleBarSpotLabelEl = document.getElementById('scale-bar-spot-label');
const scaleBarDonutLineEl = document.getElementById('scale-bar-donut-line');
const scaleBarDonutLabelEl = document.getElementById('scale-bar-donut-label');
const loadingOverlayEl = document.getElementById('loading-overlay');
const loadingFillEl = document.getElementById('loading-fill');
const loadingTextEl = document.getElementById('loading-text');
const copyDiagnosticsBtn = document.getElementById('copy-diagnostics');
const copyDiagnosticsStatusEl = document.getElementById('copy-diagnostics-status');
const zoomInBtn = document.getElementById('zoom-in');
const zoomOutBtn = document.getElementById('zoom-out');
const zoomResetBtn = document.getElementById('zoom-reset');
const rangeExpInputEl = document.getElementById('range-exp-input');
const fwhmExpInputEl = document.getElementById('fwhm-exp-input');
const useDofInputEl = document.getElementById('use-dof-input');
const nkeepInputEl = document.getElementById('nkeep-input');
const dzFieldMaxInputEl = document.getElementById('dz-field-max-input');
const dzPupilMaxInputEl = document.getElementById('dz-pupil-max-input');
const fwhmSourceSelectEl = document.getElementById('fwhm-source-select');
const sensSourceSelectEl = document.getElementById('sens-source-select');
const SCALE_BAR_WORLD_UNITS = 0.1;
const GRID_PITCH_WORLD_UNITS = 0.048;
const GRID_MAJOR_EVERY = 5;
const GRID_MINOR_COLOR = 'rgba(122, 178, 255, 0.04)';
const GRID_MAJOR_COLOR = 'rgba(122, 178, 255, 0.05)';
const CORNER_OVERLAY_POSITIONS = [
  {left: '4%', top: '80%'},
  {left: '4%', top: '8%'},
  {right: '4%', top: '80%'},
  {right: '4%', top: '8%'},
];
const ZERNIKE_DISPLAY_ROWS = [
  {label: 'Z4:', terms: [4]},
  {label: 'Z5,6:', terms: [5, 6]},
  {label: 'Z7,8:', terms: [7, 8]},
  {label: 'Z9,10:', terms: [9, 10]},
  {label: 'Z11:', terms: [11]},
  {label: 'Z12,13:', terms: [12, 13]},
  {label: 'Z14,15:', terms: [14, 15]},
  {label: 'Z16,17:', terms: [16, 17]},
  {label: 'Z18,19:', terms: [18, 19]},
  {label: 'Z20,21:', terms: [20, 21]},
  {label: 'Z22:', terms: [22]},
  {label: 'Z23,24:', terms: [23, 24]},
  {label: 'Z25,26:', terms: [25, 26]},
  {label: 'Z27,28:', terms: [27, 28]},
];
let sliderInputs = [];
let parameterInputs = [];
let currentViewState = { target: [0, 0, 0], zoom: Number.NaN };
let deckgl = null;
let lastStartupError = '';
let zernikeCornerEls = [];
let includeIntrinsics = true;
let isDraggingView = false;
let dragLastClientX = 0;
let dragLastClientY = 0;

function initializeVmodeControlDefaults() {
  if (rangeExpInputEl) rangeExpInputEl.value = String(DEFAULT_RANGE_NORM_EXP);
  if (fwhmExpInputEl) fwhmExpInputEl.value = String(DEFAULT_FWHM_NORM_EXP);
  if (useDofInputEl) useDofInputEl.value = DEFAULT_USE_DOF;
  if (nkeepInputEl) nkeepInputEl.value = String(DEFAULT_NKEEP);
  if (dzFieldMaxInputEl) dzFieldMaxInputEl.value = String(DEFAULT_DZ_FIELD_MAX);
  if (dzPupilMaxInputEl) dzPupilMaxInputEl.value = String(DEFAULT_DZ_PUPIL_MAX);
}

// Populate the source dropdowns from what the loaded model actually shipped,
// so a single-source model shows one (disabled) option rather than offering a
// choice that silently does nothing.
function populateSourceSelects() {
  const fill = (el, available, selected) => {
    if (!el) return;
    el.textContent = '';
    for (const src of WEIGHT_SOURCES) {
      if (!available[src]) continue;
      const opt = document.createElement('option');
      opt.value = src;
      opt.textContent = SOURCE_LABELS[src] || src;
      el.appendChild(opt);
    }
    el.value = selected;
    el.disabled = el.options.length < 2;
  };
  fill(fwhmSourceSelectEl, fwhmWeightsBySource, fwhmSource);
  fill(sensSourceSelectEl, wfSensBySource, sensSource);
}

function setControlsReady(ready) {
  if (!resetControlsBtn) return;
  resetControlsBtn.disabled = !ready;
  resetControlsBtn.textContent = ready ? 'Reset (r)' : 'Loading…';
}

function formatMiB(bytes) {
  return `${(bytes / (1024 * 1024)).toFixed(1)} MiB`;
}

function updateLoadingProgress(loadedBytes, totalBytes, done) {
  if (!loadingOverlayEl) return;
  if (!Number.isFinite(loadedBytes) || loadedBytes < 0) loadedBytes = 0;

  const hasTotal = Number.isFinite(totalBytes) && totalBytes > 0;
  const ratio = hasTotal ? Math.min(1, loadedBytes / totalBytes) : 0;

  if (loadingFillEl) {
    loadingFillEl.style.width = `${(ratio * 100).toFixed(1)}%`;
  }

  if (loadingTextEl) {
    if (hasTotal) {
      const percent = Math.min(100, ratio * 100).toFixed(1);
      loadingTextEl.textContent = `${percent}% (${formatMiB(loadedBytes)} / ${formatMiB(totalBytes)})`;
    } else {
      loadingTextEl.textContent = `Downloaded ${formatMiB(loadedBytes)}…`;
    }
  }

  if (done) {
    if (loadingFillEl) loadingFillEl.style.width = '100%';
    loadingOverlayEl.classList.add('is-hidden');
  }
}

function showLoadingError(errorMessage) {
  if (!loadingOverlayEl || !loadingTextEl) return;
  if (loadingFillEl) loadingFillEl.style.width = '0%';
  loadingTextEl.textContent = errorMessage;
}

function setCopyDiagnosticsStatus(message) {
  if (!copyDiagnosticsStatusEl) return;
  copyDiagnosticsStatusEl.textContent = message;
}

function zernikeIndex(cornerIndex, termIndex, dofIndex) {
  return (cornerIndex * DONUT_ZERNIKE_TERMS + termIndex) * K + dofIndex;
}

function evaluateCornerZernike(cornerIndex, termIndex) {
  const base = includeIntrinsics ? zk0[cornerIndex * DONUT_ZERNIKE_TERMS + termIndex] : 0;
  let delta = 0;
  for (let k = 0; k < K; k++) {
    delta += dZk[zernikeIndex(cornerIndex, termIndex, k)] * dofValueToModelUnits(k, p[k]);
  }
  return base + delta;
}

function formatZernikeFixed5(value) {
  if (!Number.isFinite(value)) return '-----';
  const rounded = Math.round(value);
  if (rounded < -9999 || rounded > 99999) return '#####';
  return `${rounded}`.padStart(5, ' ');
}

function formatZernikeRowText(rowLabel, rowTerms, rowValuesByTerm) {
  const label = rowLabel.padEnd(8, ' ');
  const blank = ' '.repeat(5);

  if (rowTerms.length === 1) {
    const center = formatZernikeFixed5(rowValuesByTerm[rowTerms[0]]);
    return `${label} ${blank} ${center} ${blank}`;
  }

  const left = formatZernikeFixed5(rowValuesByTerm[rowTerms[0]]);
  const right = formatZernikeFixed5(rowValuesByTerm[rowTerms[1]]);
  return `${label} ${left} ${blank} ${right}`;
}

function buildZernikeOutputs() {
  const root = document.getElementById('zernike-overlay');
  if (!root) return;

  root.innerHTML = '';
  zernikeCornerEls = [];

  for (let cornerIndex = 0; cornerIndex < DONUT_CORNERS; cornerIndex++) {
    const card = document.createElement('div');
    card.className = 'zernike-corner';
    const anchor = CORNER_OVERLAY_POSITIONS[cornerIndex] || {left: '50%', top: '50%'};
    if (anchor.left != null) card.style.left = anchor.left;
    if (anchor.right != null) card.style.right = anchor.right;
    card.style.top = anchor.top;

    const rowsEl = document.createElement('div');
    rowsEl.className = 'zernike-rows';

    const cornerRowEls = [];
    for (const rowSpec of ZERNIKE_DISPLAY_ROWS) {
      const rowEl = document.createElement('div');
      rowEl.className = 'zernike-row';

      const valueEl = document.createElement('span');
      valueEl.className = 'zernike-value';
      valueEl.textContent = formatZernikeRowText(rowSpec.label, rowSpec.terms, {});
      rowEl.appendChild(valueEl);

      rowsEl.appendChild(rowEl);
      cornerRowEls.push({label: rowSpec.label, terms: rowSpec.terms, valueEl});
    }

    card.appendChild(rowsEl);
    root.appendChild(card);
    zernikeCornerEls.push(cornerRowEls);
  }
}

function updateZernikeOutputs() {
  if (!zernikeCornerEls.length) return;

  for (let cornerIndex = 0; cornerIndex < DONUT_CORNERS; cornerIndex++) {
    const cornerRows = zernikeCornerEls[cornerIndex];
    if (!cornerRows) continue;

    for (const row of cornerRows) {
      const rowValuesByTerm = {};
      for (const term of row.terms) {
        rowValuesByTerm[term] = evaluateCornerZernike(cornerIndex, term);
      }
      row.valueEl.textContent = formatZernikeRowText(row.label, row.terms, rowValuesByTerm);
    }
  }
}

function initializeDeck() {
  if (!window.deck) {
    throw new Error('Visualization library failed to load (deck.gl script unavailable).');
  }

  DeckGL = window.deck.DeckGL;
  OrthographicView = window.deck.OrthographicView;
  ScatterplotLayer = window.deck.ScatterplotLayer;
  COORDINATE_SYSTEM = window.deck.COORDINATE_SYSTEM;

  if (!DeckGL || !OrthographicView || !ScatterplotLayer || !COORDINATE_SYSTEM) {
    throw new Error('Visualization library loaded incompletely.');
  }

  deckgl = new DeckGL({
    canvas,
    views: [new OrthographicView({id: 'ortho'})],
    initialViewState: { target: [0, 0, 0], zoom: 0 },
    controller: false,
    layers: []
  });
}

function getCompatibilityIssues() {
  const issues = [];

  if (typeof window.fetch !== 'function') issues.push('missing Fetch API');
  if (typeof window.Promise !== 'function') issues.push('missing Promise support');
  if (typeof window.Float32Array !== 'function') issues.push('missing typed array support');
  if (typeof window.requestAnimationFrame !== 'function') issues.push('missing animation frame support');
  if (!canvas || typeof canvas.getContext !== 'function') {
    issues.push('missing canvas support');
    return issues;
  }

  const gl = canvas.getContext('webgl2') || canvas.getContext('webgl');
  if (!gl) issues.push('WebGL unavailable');

  return issues;
}

function buildDiagnosticsText() {
  const issues = getCompatibilityIssues();
  const navigatorInfo = typeof navigator !== 'undefined' ? navigator : null;
  const locationInfo = typeof window !== 'undefined' ? window.location : null;
  const webgl2Supported = canvas && canvas.getContext ? !!canvas.getContext('webgl2') : false;
  const webglSupported = canvas && canvas.getContext
    ? (!!canvas.getContext('webgl') || !!canvas.getContext('experimental-webgl'))
    : false;

  const rows = [
    `timestamp_utc: ${new Date().toISOString()}`,
    `user_agent: ${navigatorInfo && navigatorInfo.userAgent ? navigatorInfo.userAgent : 'unknown'}`,
    `language: ${navigatorInfo && navigatorInfo.language ? navigatorInfo.language : 'unknown'}`,
    `platform: ${navigatorInfo && navigatorInfo.platform ? navigatorInfo.platform : 'unknown'}`,
    `url: ${locationInfo && locationInfo.href ? locationInfo.href : 'unknown'}`,
    `protocol: ${locationInfo && locationInfo.protocol ? locationInfo.protocol : 'unknown'}`,
    `deck_global_present: ${!!window.deck}`,
    `fetch_supported: ${typeof window.fetch === 'function'}`,
    `promise_supported: ${typeof window.Promise === 'function'}`,
    `typedarray_supported: ${typeof window.Float32Array === 'function'}`,
    `raf_supported: ${typeof window.requestAnimationFrame === 'function'}`,
    `webgl2_supported: ${webgl2Supported}`,
    `webgl_supported: ${webglSupported}`,
    `compatibility_issues: ${issues.length ? issues.join('; ') : 'none'}`,
    `startup_started_utc: ${loadTelemetry.startupStartIso || 'unknown'}`,
    `startup_finished_utc: ${loadTelemetry.startupEndIso || 'unknown'}`,
    `startup_duration_ms: ${formatDuration(loadTelemetry.startupDurationMs)}`,
    `meta_http_status: ${loadTelemetry.metaHttpStatus != null ? loadTelemetry.metaHttpStatus : 'unknown'}`,
    `meta_fetch_duration_ms: ${formatDuration(loadTelemetry.metaFetchDurationMs)}`,
    `meta_content_length_bytes: ${loadTelemetry.metaBytes != null ? loadTelemetry.metaBytes : 'unknown'}`,
    `model_http_status: ${loadTelemetry.modelHttpStatus != null ? loadTelemetry.modelHttpStatus : 'unknown'}`,
    `model_fetch_duration_ms: ${formatDuration(loadTelemetry.modelFetchDurationMs)}`,
    `model_content_length_bytes: ${loadTelemetry.modelBytesExpected != null ? loadTelemetry.modelBytesExpected : 'unknown'}`,
    `model_bytes_received: ${loadTelemetry.modelBytesReceived != null ? loadTelemetry.modelBytesReceived : 'unknown'}`,
    `model_streaming_reader_used: ${loadTelemetry.modelUsedStreamReader != null ? loadTelemetry.modelUsedStreamReader : 'unknown'}`,
    `model_download_completed: ${loadTelemetry.modelDownloadCompleted}`,
    `last_startup_error: ${lastStartupError || 'none'}`,
  ];

  return rows.join('\n');
}

async function copyDiagnosticsToClipboard() {
  const text = buildDiagnosticsText();

  if (navigator && navigator.clipboard && typeof navigator.clipboard.writeText === 'function') {
    await navigator.clipboard.writeText(text);
    return true;
  }

  return false;
}

function buildLoadErrorMessage(error) {
  const raw = error && error.message ? error.message : String(error);

  if (/missing|unavailable|unsupported|WebGL/i.test(raw)) {
    return `Browser compatibility error: ${raw}. Please update your browser and ensure hardware acceleration is enabled.`;
  }

  if (/deck\.gl|Visualization library/i.test(raw)) {
    return `${raw} This can happen if the CDN script is blocked by a network policy, ad blocker, or privacy extension.`;
  }

  if (/Failed to fetch|NetworkError|Load failed/i.test(raw)) {
    return `Network error while downloading model files: ${raw}. Check connection quality, proxy/firewall settings, and that the page is served over HTTP(S).`;
  }

  if (/Metadata|Packed model|N\/K/i.test(raw)) {
    return `Model data error: ${raw}. The model files may be incomplete or from mismatched versions.`;
  }

  return `Load failed: ${raw}`;
}

function getFitZoom(width, height, radius) {
  const diameter = Math.max(1e-6, 2 * radius);
  const usablePixels = Math.max(1, Math.min(width, height) * VIEW_PADDING);
  const scale = usablePixels / diameter;
  return Math.log2(scale);
}

function getFitViewState() {
  const r = vis.getBoundingClientRect();
  const zoom = getFitZoom(r.width, r.height, modelRadius);
  return {target: [0, 0, 0], zoom};
}

function applyViewState() {
  const r = vis.getBoundingClientRect();
  drawGrid(r.width, r.height);
  updateScaleBar();
  if (deckgl) {
    deckgl.setProps({ viewState: currentViewState });
  }
}

function setZoom(nextZoom) {
  const clampedZoom = Math.max(-4, Math.min(18, nextZoom));
  currentViewState = {
    target: currentViewState.target,
    zoom: clampedZoom
  };
  applyViewState();
}

function zoomBy(delta) {
  setZoom(currentViewState.zoom + delta);
}

function resetView() {
  currentViewState = getFitViewState();
  applyViewState();
}

function panByPixels(deltaX, deltaY) {
  const scale = Math.pow(2, currentViewState.zoom);
  const targetX = currentViewState.target[0] - deltaX / scale;
  const targetY = currentViewState.target[1] - deltaY / scale;
  currentViewState = {
    target: [targetX, targetY, 0],
    zoom: currentViewState.zoom
  };
  applyViewState();
}

function resizeDeck() {
  const r = vis.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  if (gridCanvas) {
    gridCanvas.width = Math.max(1, Math.floor(r.width * dpr));
    gridCanvas.height = Math.max(1, Math.floor(r.height * dpr));
  }
  canvas.width  = Math.max(1, Math.floor(r.width  * dpr));
  canvas.height = Math.max(1, Math.floor(r.height * dpr));
  if (!Number.isFinite(currentViewState.zoom)) {
    currentViewState = getFitViewState();
  }
  drawGrid(r.width, r.height);
  updateScaleBar();
  if (deckgl) {
    deckgl.setProps({
      width: r.width,
      height: r.height,
      viewState: currentViewState
    });
  }
}

function drawGrid(width, height) {
  if (!gridCanvas) return;
  const context = gridCanvas.getContext('2d');
  if (!context) return;

  const dpr = window.devicePixelRatio || 1;
  const canvasWidth = Math.max(1, Math.floor(width * dpr));
  const canvasHeight = Math.max(1, Math.floor(height * dpr));

  context.setTransform(1, 0, 0, 1, 0, 0);
  context.clearRect(0, 0, canvasWidth, canvasHeight);

  const scale = Math.pow(2, currentViewState.zoom);
  const pitchDevicePx = Math.max(Math.round(2 * dpr), Math.round(GRID_PITCH_WORLD_UNITS * scale * dpr));
  const majorPitchDevicePx = pitchDevicePx * GRID_MAJOR_EVERY;

  const originScreenDeviceX = (width / 2 + (0 - currentViewState.target[0]) * scale) * dpr;
  const originScreenDeviceY = (height / 2 - (0 - currentViewState.target[1]) * scale) * dpr;

  const offsetDeviceX = ((originScreenDeviceX % pitchDevicePx) + pitchDevicePx) % pitchDevicePx;
  const offsetDeviceY = ((originScreenDeviceY % pitchDevicePx) + pitchDevicePx) % pitchDevicePx;
  const majorOffsetDeviceX = ((originScreenDeviceX % majorPitchDevicePx) + majorPitchDevicePx) % majorPitchDevicePx;
  const majorOffsetDeviceY = ((originScreenDeviceY % majorPitchDevicePx) + majorPitchDevicePx) % majorPitchDevicePx;

  context.lineWidth = 1;

  context.strokeStyle = GRID_MINOR_COLOR;
  context.beginPath();
  for (let ix = 0, x = offsetDeviceX; x <= canvasWidth; ix++, x += pitchDevicePx) {
    if (ix % GRID_MAJOR_EVERY === 0) continue;
    const xf = Math.round(x) + 0.5;
    context.moveTo(xf, 0);
    context.lineTo(xf, canvasHeight);
  }
  for (let iy = 0, y = offsetDeviceY; y <= canvasHeight; iy++, y += pitchDevicePx) {
    if (iy % GRID_MAJOR_EVERY === 0) continue;
    const yf = Math.round(y) + 0.5;
    context.moveTo(0, yf);
    context.lineTo(canvasWidth, yf);
  }
  context.stroke();

  context.strokeStyle = GRID_MAJOR_COLOR;
  context.beginPath();
  for (let x = majorOffsetDeviceX; x <= canvasWidth; x += majorPitchDevicePx) {
    const xf = Math.round(x) + 0.5;
    context.moveTo(xf, 0);
    context.lineTo(xf, canvasHeight);
  }
  for (let y = majorOffsetDeviceY; y <= canvasHeight; y += majorPitchDevicePx) {
    const yf = Math.round(y) + 0.5;
    context.moveTo(0, yf);
    context.lineTo(canvasWidth, yf);
  }
  context.stroke();
}

function updateScaleBar() {
  const pxPerWorldUnit = Math.pow(2, currentViewState.zoom);

  if (scaleBarLineEl) {
    const widthPx = Math.max(1, SCALE_BAR_WORLD_UNITS * pxPerWorldUnit);
    scaleBarLineEl.style.width = `${widthPx.toFixed(1)}px`;
  }

  if (scaleBarSpotLineEl) {
    const spotWorldUnits = SCALE_BAR_SPOT_PIXELS * MICRONS_PER_PIXEL * spotScale;
    const widthPx = Math.max(1, spotWorldUnits * pxPerWorldUnit);
    scaleBarSpotLineEl.style.width = `${widthPx.toFixed(1)}px`;
  }
  if (scaleBarSpotLabelEl) {
    scaleBarSpotLabelEl.textContent = `${SCALE_BAR_SPOT_PIXELS} px (spot)`;
  }

  if (scaleBarDonutLineEl) {
    const donutWorldUnits = SCALE_BAR_DONUT_PIXELS * MICRONS_PER_PIXEL * donutSpotScale;
    const widthPx = Math.max(1, donutWorldUnits * pxPerWorldUnit);
    scaleBarDonutLineEl.style.width = `${widthPx.toFixed(1)}px`;
  }
  if (scaleBarDonutLabelEl) {
    scaleBarDonutLabelEl.textContent = `${SCALE_BAR_DONUT_PIXELS} px (donut)`;
  }
}

function updateMouseCoords(event) {
  if (!mouseCoordsEl) return;
  const rect = vis.getBoundingClientRect();
  const px = event.clientX - rect.left;
  const py = event.clientY - rect.top;
  const scale = Math.pow(2, currentViewState.zoom);
  const wx = (px - rect.width / 2) / scale + currentViewState.target[0];
  const wy = (rect.height / 2 - py) / scale + currentViewState.target[1];
  mouseCoordsEl.textContent = `x: ${wx.toFixed(3)}, y: ${wy.toFixed(3)}`;
}

function clearMouseCoords() {
  if (!mouseCoordsEl) return;
  mouseCoordsEl.textContent = 'x: ---, y: ---';
}

let posVersion = 0;

function render() {
  if (!deckgl || !ScatterplotLayer || !COORDINATE_SYSTEM) return;

  const layer = new ScatterplotLayer({
    id: 'pts',
    data,
    coordinateSystem: COORDINATE_SYSTEM.CARTESIAN,
    getPosition: (i) => [positions[2*i], positions[2*i + 1], 0],
    updateTriggers: {
      getPosition: posVersion
    },
    getRadius: 1.0,
    radiusUnits: 'pixels',
    stroked: false,
    getFillColor: [120, 180, 255, 60]
  });

  deckgl.setProps({ layers: [layer] });
}

let pending = false;
function requestUpdate() {
  if (pending) return;
  pending = true;
  requestAnimationFrame(() => {
    pending = false;
    updatePositions();
    posVersion++;
    render();
    updateZernikeOutputs();
  });
}

function buildSliders() {
  const root = document.getElementById('sliders');
  root.innerHTML = '';
  sliderInputs = [];
  parameterInputs = [];

  function applyControlValue(controlIndex, rawValue) {
    const parsed = Number(rawValue);
    if (!Number.isFinite(parsed)) return;
    const clamped = clampControlValue(controlIndex, parsed);
    p[controlIndex] = clamped;

    if (sliderInputs[controlIndex]) {
      sliderInputs[controlIndex].value = String(clamped);
    }
    if (parameterInputs[controlIndex]) {
      parameterInputs[controlIndex].value = formatControlValue(clamped);
    }

    requestUpdate();
  }

  function commitNumericInput(controlIndex) {
    const input = parameterInputs[controlIndex];
    if (!input) return;

    const raw = input.value.trim();
    if (raw === '') {
      input.value = formatControlValue(p[controlIndex]);
      return;
    }

    const parsed = Number(raw);
    if (!Number.isFinite(parsed)) {
      input.value = formatControlValue(p[controlIndex]);
      return;
    }

    applyControlValue(controlIndex, parsed);
  }

  for (let k = 0; k < K; k++) {
    const spec = getControlSpec(k);
    const row = document.createElement('div');
    row.className = 'row';

    const label = document.createElement('label');
    label.textContent = spec.label;

    const input = document.createElement('input');
    input.type = 'range';
    input.min = String(-spec.range);
    input.max = String(spec.range);
    input.step = String(spec.step);
    input.value = '0';

    const numericInput = document.createElement('input');
    numericInput.type = 'number';
    numericInput.min = String(-spec.range);
    numericInput.max = String(spec.range);
    numericInput.step = String(spec.step);
    numericInput.value = formatControlValue(0);

    input.addEventListener('input', () => {
      applyControlValue(k, input.value);
    });

    numericInput.addEventListener('keydown', (event) => {
      if (event.key === 'Enter') {
        event.preventDefault();
        commitNumericInput(k);
        return;
      }

      if (event.key === 'ArrowUp' || event.key === 'ArrowDown') {
        requestAnimationFrame(() => {
          commitNumericInput(k);
        });
      }
    });

    numericInput.addEventListener('change', () => {
      commitNumericInput(k);
    });

    numericInput.addEventListener('blur', () => {
      commitNumericInput(k);
    });

    row.appendChild(label);
    row.appendChild(input);
    row.appendChild(numericInput);
    root.appendChild(row);
    sliderInputs.push(input);
    parameterInputs.push(numericInput);
  }
}

function resetControls() {
  for (let k = 0; k < K; k++) {
    p[k] = 0;
    if (sliderInputs[k]) sliderInputs[k].value = '0';
    if (parameterInputs[k]) parameterInputs[k].value = formatControlValue(0);
  }
  for (let m = 0; m < vValues.length; m++) {
    vValues[m] = 0;
    if (vmodeSliderInputs[m]) vmodeSliderInputs[m].value = '0';
    if (vmodeParamInputs[m]) vmodeParamInputs[m].value = formatControlValue(0);
  }

  rebuildVmodeUI();
  requestUpdate();
}

function buildNormVector() {
  if (!rangeWeights || !fwhmWeights || rangeWeights.length !== K || fwhmWeights.length !== K) {
    return null;
  }
  const out = new Float64Array(K);
  for (let i = 0; i < K; i++) {
    out[i] = Math.pow(rangeWeights[i], rangeNormExp) * Math.pow(fwhmWeights[i], fwhmNormExp);
  }
  return out;
}

function recomputeVmodes() {
  const rangeExpInput = document.getElementById('range-exp-input');
  const fwhmExpInput = document.getElementById('fwhm-exp-input');
  const useDofInput = document.getElementById('use-dof-input');
  const nkeepInput = document.getElementById('nkeep-input');
  if (!rangeExpInput || !fwhmExpInput || !useDofInput || !nkeepInput) return;

  const parsedRangeExp = Number(rangeExpInput.value);
  const parsedFwhmExp = Number(fwhmExpInput.value);
  rangeNormExp = Number.isFinite(parsedRangeExp) ? parsedRangeExp : DEFAULT_RANGE_NORM_EXP;
  fwhmNormExp = Number.isFinite(parsedFwhmExp) ? parsedFwhmExp : DEFAULT_FWHM_NORM_EXP;
  rangeExpInput.value = String(rangeNormExp);
  fwhmExpInput.value = String(fwhmNormExp);

  // Pick up the FWHM-weight and sensitivity source selections.  Both feed the
  // SVD below, so changing either invalidates every v-mode.
  const fwhmSourceEl = document.getElementById('fwhm-source-select');
  if (fwhmSourceEl && fwhmWeightsBySource[fwhmSourceEl.value]) {
    fwhmSource = fwhmSourceEl.value;
  }
  const sensSourceEl = document.getElementById('sens-source-select');
  if (sensSourceEl && wfSensBySource[sensSourceEl.value]) {
    sensSource = sensSourceEl.value;
  }
  if (fwhmWeightsBySource[fwhmSource]) {
    fwhmWeights = new Float64Array(fwhmWeightsBySource[fwhmSource]);
  }
  if (wfSensBySource[sensSource]) {
    wfSens = wfSensBySource[sensSource];
  }

  currentUseDof = parseUseDof(useDofInput.value).filter((dof) => dof >= 0 && dof < K);
  useDofInput.value = formatUseDof(currentUseDof);
  nActive = currentUseDof.length;
  if (nActive === 0) {
    currentNkeep = 0;
    nkeepInput.value = '0';
  } else {
    currentNkeep = Math.max(1, Math.min(nActive, Number(nkeepInput.value) || DEFAULT_NKEEP));
    nkeepInput.value = String(currentNkeep);
  }

  norm = buildNormVector();

  if (norm && wfSens && wfSensRows > 0 && nActive > 0) {
    const result = computeVh(wfSens, wfSensRows, K, norm, currentUseDof, currentNkeep);
    Vh = result.vh;
    fullMixMatrix = result.fullMatrix;
    singularValues = result.singularValues;
  } else {
    Vh = null;
    fullMixMatrix = null;
    singularValues = null;
  }

  rebuildVmodeUI();
}

function rebuildVmodeUI() {
  vValues = new Array(currentNkeep).fill(0);
  buildVmodeSliders();
  rebuildVmodeDropdown();
  drawMixingMatrix();
  drawDzSensPlot();
  drawDzVmodePlot();
}

function rebuildVmodeDropdown() {
  const selectEl = document.getElementById('dz-vmode-select');
  if (!selectEl) return;
  const prevValue = Number(selectEl.value) || 0;
  selectEl.innerHTML = '';
  const count = currentNkeep > 0 ? currentNkeep : nActive;
  for (let m = 0; m < count; m++) {
    const opt = document.createElement('option');
    opt.value = String(m);
    opt.textContent = `V${m + 1}`;
    selectEl.appendChild(opt);
  }
  // Restore previous selection if still valid
  const clampedPrev = Math.max(0, Math.min(count - 1, prevValue));
  selectEl.value = String(clampedPrev);
}

function buildVmodeSliders() {
  const root = document.getElementById('vmode-sliders');
  if (!root) return;
  root.innerHTML = '';
  vmodeSliderInputs = [];
  vmodeParamInputs = [];

  if (!Vh || currentNkeep <= 0) return;

  // Compute feasible per-mode delta bounds so applying this mode alone
  // cannot drive any active DOF outside its configured range.
  const vmodeMins = new Float64Array(currentNkeep);
  const vmodeMaxs = new Float64Array(currentNkeep);
  for (let m = 0; m < currentNkeep; m++) {
    const rowOff = m * nActive;
    let lower = -Infinity;
    let upper = Infinity;
    for (let j = 0; j < nActive; j++) {
      const dofIdx = currentUseDof[j];
      const coeff = Vh[rowOff + j];
      const dofRange = CONTROL_RANGES[dofIdx] != null ? CONTROL_RANGES[dofIdx] : Infinity;
      const p0 = p[dofIdx];
      if (Math.abs(coeff) <= 1e-15 || !isFinite(dofRange)) {
        continue;
      }

      // p0 + delta*coeff must stay in [-dofRange, +dofRange]
      const a = (-dofRange - p0) / coeff;
      const b = (dofRange - p0) / coeff;
      const thisLow = Math.min(a, b);
      const thisHigh = Math.max(a, b);
      if (thisLow > lower) lower = thisLow;
      if (thisHigh < upper) upper = thisHigh;
    }

    if (!isFinite(lower) || !isFinite(upper) || upper <= lower) {
      lower = -1;
      upper = 1;
    }
    vmodeMins[m] = lower;
    vmodeMaxs[m] = upper;
  }

  for (let m = 0; m < currentNkeep; m++) {
    const row = document.createElement('div');
    row.className = 'row';

    const label = document.createElement('label');
    label.textContent = `V${m + 1}`;

    const modeMin = vmodeMins[m];
    const modeMax = vmodeMaxs[m];
    const VMODE_STEP = Math.max((modeMax - modeMin) / 400, 1e-6);

    const slider = document.createElement('input');
    slider.type = 'range';
    slider.min = String(modeMin);
    slider.max = String(modeMax);
    slider.step = String(VMODE_STEP);
    slider.value = '0';

    const numInput = document.createElement('input');
    numInput.type = 'number';
    numInput.min = String(modeMin);
    numInput.max = String(modeMax);
    numInput.step = String(VMODE_STEP);
    numInput.value = formatControlValue(0);

    const modeIndex = m; // capture for closure

    function applyVmodeDelta(newVal) {
      const clamped = Math.max(modeMin, Math.min(modeMax, newVal));
      const delta = clamped - vValues[modeIndex];
      vValues[modeIndex] = clamped;

      // Propagate delta into DOF sliders: p[useDof[j]] += delta * Vh[mode][j]
      if (Vh && Math.abs(delta) > 1e-15) {
        const rowOff = modeIndex * nActive;
        for (let j = 0; j < nActive; j++) {
          const dofIdx = currentUseDof[j];
          p[dofIdx] += delta * Vh[rowOff + j];
          // Update DOF slider display
          const clamped2 = clampControlValue(dofIdx, p[dofIdx]);
          p[dofIdx] = clamped2;
          if (sliderInputs[dofIdx]) sliderInputs[dofIdx].value = String(clamped2);
          if (parameterInputs[dofIdx]) parameterInputs[dofIdx].value = formatControlValue(clamped2);
        }
      }

      // Sync vmode slider/input displays
      if (vmodeSliderInputs[modeIndex]) vmodeSliderInputs[modeIndex].value = String(clamped);
      if (vmodeParamInputs[modeIndex]) vmodeParamInputs[modeIndex].value = formatControlValue(clamped);

      requestUpdate();
    }

    slider.addEventListener('input', () => {
      applyVmodeDelta(Number(slider.value));
    });

    function commitVmodeInput() {
      const raw = numInput.value.trim();
      if (raw === '') {
        numInput.value = formatControlValue(vValues[modeIndex]);
        return;
      }
      const parsed = Number(raw);
      if (!Number.isFinite(parsed)) {
        numInput.value = formatControlValue(vValues[modeIndex]);
        return;
      }
      applyVmodeDelta(parsed);
    }

    numInput.addEventListener('keydown', (event) => {
      if (event.key === 'Enter') { event.preventDefault(); commitVmodeInput(); return; }
      if (event.key === 'Escape') { event.preventDefault(); numInput.blur(); return; }
      if (event.key === 'ArrowUp' || event.key === 'ArrowDown') {
        requestAnimationFrame(() => commitVmodeInput());
      }
    });
    numInput.addEventListener('change', commitVmodeInput);
    numInput.addEventListener('blur', commitVmodeInput);

    row.appendChild(label);
    row.appendChild(slider);
    row.appendChild(numInput);
    root.appendChild(row);
    vmodeSliderInputs.push(slider);
    vmodeParamInputs.push(numInput);
  }
}

function bwrColor(t) {
  t = Math.max(-1, Math.min(1, t));
  let r, g, b;
  if (t < 0) {
    const s = 1 + t;
    r = Math.round(255 * s);
    g = Math.round(255 * s);
    b = 255;
  } else {
    const s = 1 - t;
    r = 255;
    g = Math.round(255 * s);
    b = Math.round(255 * s);
  }
  return `rgb(${r},${g},${b})`;
}

function computeGroupBoundaries(useDof) {
  const m2Hex = [], camHex = [], m1m3Bend = [], m2Bend = [];
  for (let i = 0; i < useDof.length; i++) {
    const d = useDof[i];
    if (d < 5) m2Hex.push(i);
    else if (d < 10) camHex.push(i);
    else if (d < 30) m1m3Bend.push(i);
    else m2Bend.push(i);
  }
  const boundaries = [];
  if (m2Hex.length > 0 && (camHex.length + m1m3Bend.length + m2Bend.length) > 0)
    boundaries.push(m2Hex[m2Hex.length - 1]);
  if (camHex.length > 0 && (m1m3Bend.length + m2Bend.length) > 0)
    boundaries.push(camHex[camHex.length - 1]);
  if (m1m3Bend.length > 0 && m2Bend.length > 0)
    boundaries.push(m1m3Bend[m1m3Bend.length - 1]);
  return boundaries;
}

function drawModeXAxisTicks(ctx, plotLeft, y, nModes, cellSize, step) {
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  for (let m = 0; m < nModes; m++) {
    if (step > 1 && m % step !== 0 && m !== nModes - 1) continue;
    ctx.fillText(String(m + 1), plotLeft + m * cellSize + cellSize / 2, y);
  }
}

function drawMixingMatrix() {
  const canvas = document.getElementById('mixing-matrix-canvas');
  if (!canvas) return;

  if (!fullMixMatrix || !singularValues || nActive <= 0) {
    hideCanvas(canvas);
    return;
  }

  const CELL = 14;
  const LEFT_PAD = 18;
  const LABEL_W = 75;
  const SV_H = 84;
  const SV_GAP = 8;
  const BOTTOM_H = 32;
  const TOP_PAD = 2;
  const nModes = nActive;
  const nDofs = nActive;
  const matrixTop = TOP_PAD + SV_H + SV_GAP;

  const canvasW = LEFT_PAD + LABEL_W + nModes * CELL;
  const canvasH = matrixTop + nDofs * CELL + BOTTOM_H;
  const ctx = setupHiDpiCanvas(canvas, canvasW, canvasH);

  // Normalize: mix @ diag(1/norm[use_dof])
  const normMix = new Float64Array(fullMixMatrix.length);
  for (let mode = 0; mode < nModes; mode++) {
    for (let j = 0; j < nDofs; j++) {
      const dofIdx = currentUseDof[j];
      const n = (norm && norm[dofIdx]) ? norm[dofIdx] : 1;
      normMix[mode * nActive + j] = fullMixMatrix[mode * nActive + j] / n;
    }
  }

  let vmax = 0;
  for (let i = 0; i < normMix.length; i++) {
    vmax = Math.max(vmax, Math.abs(normMix[i]));
  }
  if (vmax < 1e-15) vmax = 1;

  const plotLeft = LEFT_PAD + LABEL_W;
  const plotRight = plotLeft + nModes * CELL;
  const svTop = TOP_PAD;
  const svBottom = svTop + SV_H;
  let svMinLog = Infinity;
  let svMaxLog = -Infinity;
  for (let mode = 0; mode < nModes; mode++) {
    const sigma = singularValues[mode];
    if (sigma <= 0) continue;
    const logSigma = Math.log10(sigma);
    if (logSigma < svMinLog) svMinLog = logSigma;
    if (logSigma > svMaxLog) svMaxLog = logSigma;
  }
  if (!Number.isFinite(svMinLog) || !Number.isFinite(svMaxLog)) {
    svMinLog = 0;
    svMaxLog = 1;
  } else if (!(svMaxLog > svMinLog)) {
    svMinLog -= 0.5;
    svMaxLog += 0.5;
  }

  ctx.strokeStyle = 'rgba(0, 0, 0, 0.15)';
  ctx.lineWidth = 1;
  ctx.strokeRect(plotLeft, svTop, nModes * CELL, SV_H);

  // Y ticks at powers of ten for readability.
  const tickPowers = [];
  const minPow = Math.ceil(svMinLog);
  const maxPow = Math.floor(svMaxLog);
  if (maxPow >= minPow) {
    for (let p = minPow; p <= maxPow; p++) tickPowers.push(p);
  } else {
    tickPowers.push(Math.round(svMinLog));
  }
  if (tickPowers.length > 6) {
    const stride = Math.ceil(tickPowers.length / 6);
    const reduced = [];
    for (let i = 0; i < tickPowers.length; i += stride) reduced.push(tickPowers[i]);
    if (reduced[reduced.length - 1] !== tickPowers[tickPowers.length - 1]) {
      reduced.push(tickPowers[tickPowers.length - 1]);
    }
    tickPowers.length = 0;
    tickPowers.push(...reduced);
  }
  ctx.strokeStyle = 'rgba(0, 0, 0, 0.35)';
  ctx.fillStyle = '#333';
  ctx.font = '8px ui-monospace, Menlo, monospace';
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  for (const p of tickPowers) {
    const y = svBottom - ((p - svMinLog) / (svMaxLog - svMinLog)) * SV_H;
    ctx.beginPath();
    ctx.moveTo(plotLeft - 4, y);
    ctx.lineTo(plotLeft, y);
    ctx.stroke();
    ctx.fillText(String(p), plotLeft - 6, y);
  }

  ctx.beginPath();
  let started = false;
  for (let mode = 0; mode < nModes; mode++) {
    const sigma = singularValues[mode];
    const x = plotLeft + mode * CELL + CELL / 2;
    const y = sigma > 0
      ? svBottom - ((Math.log10(sigma) - svMinLog) / (svMaxLog - svMinLog)) * SV_H
      : svBottom;
    if (!started) {
      ctx.moveTo(x, y);
      started = true;
    } else {
      ctx.lineTo(x, y);
    }
  }
  ctx.strokeStyle = '#222';
  ctx.lineWidth = 1.5;
  ctx.stroke();

  ctx.fillStyle = '#222';
  for (let mode = 0; mode < nModes; mode++) {
    const sigma = singularValues[mode];
    const x = plotLeft + mode * CELL + CELL / 2;
    const y = sigma > 0
      ? svBottom - ((Math.log10(sigma) - svMinLog) / (svMaxLog - svMinLog)) * SV_H
      : svBottom;
    ctx.beginPath();
    ctx.arc(x, y, 1.75, 0, 2 * Math.PI);
    ctx.fill();
  }

  ctx.save();
  ctx.translate(8, svTop + SV_H / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  ctx.fillText('Log S.V.', 0, 0);
  ctx.restore();

  // Draw cells (origin="lower": DOF index 0 at bottom)
  for (let mode = 0; mode < nModes; mode++) {
    for (let j = 0; j < nDofs; j++) {
      const val = normMix[mode * nActive + j];
      ctx.fillStyle = bwrColor(val / vmax);
      ctx.fillRect(plotLeft + mode * CELL, matrixTop + (nDofs - 1 - j) * CELL, CELL, CELL);
    }
  }

  // Grey overlay for modes beyond nkeep so it stays distinct from the value colormap.
  drawUnusedModesOverlay(ctx, plotLeft, svTop, nModes, currentNkeep, CELL, SV_H);
  drawUnusedModesOverlay(ctx, plotLeft, matrixTop, nModes, currentNkeep, CELL, nDofs * CELL);

  // Group separator lines
  const bounds = computeGroupBoundaries(currentUseDof);
  ctx.strokeStyle = 'rgba(0, 0, 0, 0.3)';
  ctx.lineWidth = 1;
  for (const bIdx of bounds) {
    const y = matrixTop + (nDofs - 1 - bIdx) * CELL;
    ctx.beginPath();
    ctx.moveTo(plotLeft, y);
    ctx.lineTo(plotRight, y);
    ctx.stroke();
  }

  // Y-axis labels (DOF names)
  ctx.fillStyle = '#333';
  ctx.font = '8px ui-monospace, Menlo, monospace';
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  for (let j = 0; j < nDofs; j++) {
    const dofIdx = currentUseDof[j];
    const full = CONTROL_NAMES[dofIdx] || ('DOF ' + dofIdx);
    const name = full.replace(/\s*\[.*\]$/, '');
    ctx.fillText(name, LEFT_PAD + LABEL_W - 3, matrixTop + (nDofs - 1 - j) * CELL + CELL / 2);
  }

  // X-axis labels (mode numbers)
  drawModeXAxisTicks(ctx, plotLeft, matrixTop + nDofs * CELL + 2, nModes, CELL, 1);

  ctx.font = '10px ui-sans-serif, system-ui, sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  ctx.fillText('Vmode', plotLeft + (nModes * CELL) / 2, matrixTop + nDofs * CELL + 16);

  ctx.save();
  ctx.translate(8, matrixTop + (nDofs * CELL) / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  ctx.fillText('DOF', 0, 0);
  ctx.restore();
}

function getDzFieldMax() {
  const maxAvailable = wfSensNFieldDz > 0
    ? Math.max(1, wfSensNFieldDz - 1)
    : DEFAULT_DZ_FIELD_MAX;

  if (!dzFieldMaxInputEl) {
    return Math.min(DEFAULT_DZ_FIELD_MAX, maxAvailable);
  }

  const parsed = Math.trunc(Number(dzFieldMaxInputEl.value));
  const requested = Number.isFinite(parsed) ? parsed : DEFAULT_DZ_FIELD_MAX;
  const clamped = Math.max(1, Math.min(maxAvailable, requested));
  dzFieldMaxInputEl.value = String(clamped);
  return clamped;
}

function getDzPupilMax() {
  const maxAvailable = wfSensNPupilZk > 0
    ? Math.max(4, wfSensNPupilZk - 1)
    : DEFAULT_DZ_PUPIL_MAX;

  if (!dzPupilMaxInputEl) {
    return Math.min(DEFAULT_DZ_PUPIL_MAX, maxAvailable);
  }

  const parsed = Math.trunc(Number(dzPupilMaxInputEl.value));
  const requested = Number.isFinite(parsed) ? parsed : DEFAULT_DZ_PUPIL_MAX;
  const clamped = Math.max(4, Math.min(maxAvailable, requested));
  dzPupilMaxInputEl.value = String(clamped);
  return clamped;
}

function drawDzSensPlot() {
  const canvas = document.getElementById('dz-sens-canvas');
  if (!canvas) return;

  if (!fullMixMatrix || !wfSens || nActive <= 0 || wfSensNPupilZk <= 0 || currentNkeep <= 0) {
    hideCanvas(canvas);
    return;
  }

  // DZ rows are driven by the user-selected field and pupil limits.
  // Indices are interpreted directly in the packed sensitivity tensor.
  const fieldMax = getDzFieldMax();
  const pupilMax = getDzPupilMax();
  const FIELD_DZS = Array.from({length: fieldMax}, (_, index) => index + 1);
  const PUPIL_ZKS = Array.from({length: pupilMax - 3}, (_, index) => index + 4);

  const dzFlatRows = [];
  const dzLabels = [];
  for (const fdz of FIELD_DZS) {
    for (const pzk of PUPIL_ZKS) {
      dzFlatRows.push(fdz * wfSensNPupilZk + pzk);
      dzLabels.push(`(${fdz}, ${pzk})`);
    }
  }

  const nRows = dzFlatRows.length;
  const nModes = nActive;

  // Compute projection: proj[i, m] = sum_j wfSens[flatRow * K + useDof[j]] * fullMixMatrix[m * nActive + j]
  const proj = new Float64Array(nRows * nModes);
  for (let i = 0; i < nRows; i++) {
    const sensOff = dzFlatRows[i] * K;
    for (let m = 0; m < nModes; m++) {
      let val = 0;
      const mixOff = m * nActive;
      for (let j = 0; j < nActive; j++) {
        val += wfSens[sensOff + currentUseDof[j]] * fullMixMatrix[mixOff + j];
      }
      proj[i * nModes + m] = val;
    }
  }

  const columnMax = new Float64Array(nModes);
  for (let m = 0; m < nModes; m++) {
    let vmax = 0;
    for (let i = 0; i < nRows; i++) {
      vmax = Math.max(vmax, Math.abs(proj[i * nModes + m]));
    }
    columnMax[m] = vmax < 1e-15 ? 1 : vmax;
  }

  const CELL = 14;
  const LEFT_PAD = 18;
  const LABEL_W = 75;
  const BOTTOM_H = 32;
  const TOP_PAD = 2;

  const canvasW = LEFT_PAD + LABEL_W + nModes * CELL;
  const canvasH = TOP_PAD + nRows * CELL + BOTTOM_H;
  const ctx = setupHiDpiCanvas(canvas, canvasW, canvasH);

  // Draw cells with row index 0 at top.
  for (let i = 0; i < nRows; i++) {
    for (let m = 0; m < nModes; m++) {
      ctx.fillStyle = bwrColor(proj[i * nModes + m] / columnMax[m]);
      ctx.fillRect(LEFT_PAD + LABEL_W + m * CELL, TOP_PAD + i * CELL, CELL, CELL);
    }
  }

  // Grey overlay for modes beyond nkeep so it stays distinct from the value colormap.
  drawUnusedModesOverlay(ctx, LEFT_PAD + LABEL_W, TOP_PAD, nModes, currentNkeep, CELL, nRows * CELL);

  // Separator lines between field DZ groups
  ctx.strokeStyle = 'rgba(0, 0, 0, 0.3)';
  ctx.lineWidth = 1;
  for (let g = 1; g < FIELD_DZS.length; g++) {
    const sepIdx = g * PUPIL_ZKS.length;
    const y = TOP_PAD + sepIdx * CELL;
    ctx.beginPath();
    ctx.moveTo(LEFT_PAD + LABEL_W, y);
    ctx.lineTo(LEFT_PAD + LABEL_W + nModes * CELL, y);
    ctx.stroke();
  }

  // Y-axis labels
  ctx.fillStyle = '#333';
  ctx.font = '8px ui-monospace, Menlo, monospace';
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  for (let i = 0; i < nRows; i++) {
    ctx.fillText(dzLabels[i], LEFT_PAD + LABEL_W - 3, TOP_PAD + i * CELL + CELL / 2);
  }

  // X-axis labels (mode numbers)
  const xStep = nModes > 30 ? 5 : (nModes > 15 ? 2 : 1);
  drawModeXAxisTicks(ctx, LEFT_PAD + LABEL_W, TOP_PAD + nRows * CELL + 2, nModes, CELL, xStep);

  ctx.font = '10px ui-sans-serif, system-ui, sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  ctx.fillText('Vmode', LEFT_PAD + LABEL_W + (nModes * CELL) / 2, TOP_PAD + nRows * CELL + 16);

  ctx.save();
  ctx.translate(8, TOP_PAD + (nRows * CELL) / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  ctx.fillText('Double Zernike', 0, 0);
  ctx.restore();
}

function drawDzVmodePlot() {
  const canvas = document.getElementById('dz-vmode-canvas');
  if (!canvas) return;

  if (!fullMixMatrix || !wfSens || nActive <= 0 || wfSensNPupilZk <= 0 || currentNkeep <= 0) {
    hideCanvas(canvas);
    return;
  }

  const selectEl = document.getElementById('dz-vmode-select');
  const selectedMode = selectEl ? Math.max(0, Math.min(nActive - 1, Number(selectEl.value) || 0)) : 0;

  // Fixed ranges independent of the field/pupil max controls
  const FIELD_DZS = Array.from({length: 15}, (_, i) => i + 1);  // k = 1..15
  const PUPIL_ZKS = Array.from({length: 25}, (_, i) => i + 4);  // j = 4..28

  const nK = FIELD_DZS.length;
  const nJ = PUPIL_ZKS.length;

  // Clamp to available data
  const availK = Math.min(nK, wfSensNFieldDz > 0 ? wfSensNFieldDz - 1 : nK);
  const availJ = Math.min(nJ, wfSensNPupilZk > 0 ? wfSensNPupilZk - 1 : nJ);

  // Compute projection for only the selected mode: proj2d[ki * nJ + ji]
  const proj2d = new Float64Array(nK * nJ);
  const mixOff = selectedMode * nActive;
  for (let ki = 0; ki < availK; ki++) {
    const fdz = FIELD_DZS[ki];
    for (let ji = 0; ji < availJ; ji++) {
      const pzk = PUPIL_ZKS[ji];
      const sensOff = (fdz * wfSensNPupilZk + pzk) * K;
      let val = 0;
      for (let j = 0; j < nActive; j++) {
        val += wfSens[sensOff + currentUseDof[j]] * fullMixMatrix[mixOff + j];
      }
      proj2d[ki * nJ + ji] = val;
    }
  }

  let vmax = 0;
  for (let i = 0; i < proj2d.length; i++) {
    vmax = Math.max(vmax, Math.abs(proj2d[i]));
  }
  if (vmax < 1e-15) vmax = 1;

  const CELL = 14;
  const LEFT_PAD = 18;
  const LABEL_W = 32;
  const TOP_PAD = 2;
  const BOTTOM_H = 32;
  const RIGHT_PAD = 8;

  // rows = field DZ (k, k1 at top), cols = pupil ZK (j, j4 at left)
  const canvasW = LEFT_PAD + LABEL_W + nJ * CELL + RIGHT_PAD;
  const canvasH = TOP_PAD + nK * CELL + BOTTOM_H;
  const ctx = setupHiDpiCanvas(canvas, canvasW, canvasH);

  const plotLeft = LEFT_PAD + LABEL_W;
  const plotTop = TOP_PAD;

  // Draw cells: row=ki (field DZ k), col=ji (pupil ZK j)
  for (let ki = 0; ki < nK; ki++) {
    for (let ji = 0; ji < nJ; ji++) {
      ctx.fillStyle = bwrColor(proj2d[ki * nJ + ji] / vmax);
      ctx.fillRect(plotLeft + ji * CELL, plotTop + ki * CELL, CELL, CELL);
    }
  }

  // Cell grid lines
  ctx.strokeStyle = 'rgba(0,0,0,0.1)';
  ctx.lineWidth = 0.5;
  for (let ji = 0; ji <= nJ; ji++) {
    ctx.beginPath();
    ctx.moveTo(plotLeft + ji * CELL, plotTop);
    ctx.lineTo(plotLeft + ji * CELL, plotTop + nK * CELL);
    ctx.stroke();
  }
  for (let ki = 0; ki <= nK; ki++) {
    ctx.beginPath();
    ctx.moveTo(plotLeft, plotTop + ki * CELL);
    ctx.lineTo(plotLeft + nJ * CELL, plotTop + ki * CELL);
    ctx.stroke();
  }

  // Y-axis labels (field DZ k)
  ctx.fillStyle = '#333';
  ctx.font = '8px ui-monospace, Menlo, monospace';
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  for (let ki = 0; ki < nK; ki++) {
    ctx.fillText(String(FIELD_DZS[ki]), plotLeft - 3, plotTop + ki * CELL + CELL / 2);
  }

  // X-axis labels (pupil ZK j)
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  const xStep = nJ > 15 ? 2 : 1;
  for (let ji = 0; ji < nJ; ji += xStep) {
    ctx.fillText(String(PUPIL_ZKS[ji]), plotLeft + ji * CELL + CELL / 2, plotTop + nK * CELL + 2);
  }

  // Axis titles
  ctx.font = '10px ui-sans-serif, system-ui, sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  ctx.fillText('Pupil ZK (j)', plotLeft + (nJ * CELL) / 2, plotTop + nK * CELL + 16);

  ctx.save();
  ctx.translate(8, plotTop + (nK * CELL) / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  ctx.fillText('Field DZ (k)', 0, 0);
  ctx.restore();
}

async function start() {
  const startupStart = nowMs();
  loadTelemetry.startupStartIso = new Date().toISOString();
  loadTelemetry.startupEndIso = null;
  loadTelemetry.startupDurationMs = null;
  loadTelemetry.modelDownloadCompleted = false;
  setControlsReady(false);
  updateLoadingProgress(0, NaN, false);

  const compatibilityIssues = getCompatibilityIssues();
  if (compatibilityIssues.length) {
    throw new Error(`unsupported browser features: ${compatibilityIssues.join(', ')}`);
  }

  initializeDeck();
  const model = await loadPackedModel('./model_meta.json', './model.f32');

  initModel(model);

  buildSliders();
  populateSourceSelects();
  recomputeVmodes();
  resizeDeck();
  updatePositions();
  posVersion++;
  render();
  updateZernikeOutputs();
  setControlsReady(true);
  loadTelemetry.startupEndIso = new Date().toISOString();
  loadTelemetry.startupDurationMs = Math.max(0, nowMs() - startupStart);
}

start().catch((error) => {
  loadTelemetry.startupEndIso = new Date().toISOString();
  if (loadTelemetry.startupStartIso && !Number.isFinite(loadTelemetry.startupDurationMs)) {
    const fallbackStart = Date.parse(loadTelemetry.startupStartIso);
    if (Number.isFinite(fallbackStart)) {
      loadTelemetry.startupDurationMs = Math.max(0, Date.now() - fallbackStart);
    }
  }
  lastStartupError = error && error.message ? error.message : String(error);
  console.error(error);
  setControlsReady(false);
  showLoadingError(buildLoadErrorMessage(error));
});

if (copyDiagnosticsBtn) {
  copyDiagnosticsBtn.addEventListener('click', async () => {
    setCopyDiagnosticsStatus('');
    copyDiagnosticsBtn.disabled = true;

    try {
      const copied = await copyDiagnosticsToClipboard();
      if (copied) {
        setCopyDiagnosticsStatus('Copied');
      } else {
        const text = buildDiagnosticsText();
        window.prompt('Copy diagnostics:', text);
        setCopyDiagnosticsStatus('Opened copy dialog');
      }
    } catch (error) {
      const text = buildDiagnosticsText();
      window.prompt('Copy diagnostics:', text);
      setCopyDiagnosticsStatus('Clipboard blocked; opened dialog');
    } finally {
      copyDiagnosticsBtn.disabled = false;
    }
  });
}

window.addEventListener('resize', () => { resizeDeck(); render(); });

if (resetControlsBtn) {
  resetControlsBtn.addEventListener('click', resetControls);
}

if (includeIntrinsicsCheckbox) {
  includeIntrinsicsCheckbox.checked = includeIntrinsics;
  includeIntrinsicsCheckbox.addEventListener('change', () => {
    includeIntrinsics = includeIntrinsicsCheckbox.checked;
    updateZernikeOutputs();
  });
}

// Apply vmodes button
const applyVmodesBtn = document.getElementById('apply-vmodes');
if (applyVmodesBtn) {
  applyVmodesBtn.addEventListener('click', recomputeVmodes);
}

initializeVmodeControlDefaults();

bindEnterEscapeInput(rangeExpInputEl, recomputeVmodes);
if (rangeExpInputEl) rangeExpInputEl.addEventListener('change', recomputeVmodes);

bindEnterEscapeInput(fwhmExpInputEl, recomputeVmodes);
if (fwhmExpInputEl) fwhmExpInputEl.addEventListener('change', recomputeVmodes);

bindEnterEscapeInput(useDofInputEl, recomputeVmodes);
bindEnterEscapeInput(nkeepInputEl, recomputeVmodes);

// Switching either source rebuilds the SVD from scratch; the DOF slider values
// in `p` are untouched, so the rendered spots only change via the new v-modes.
if (fwhmSourceSelectEl) fwhmSourceSelectEl.addEventListener('change', recomputeVmodes);
if (sensSourceSelectEl) sensSourceSelectEl.addEventListener('change', recomputeVmodes);

bindEnterEscapeInput(dzFieldMaxInputEl, drawDzSensPlot);
  if (dzFieldMaxInputEl) dzFieldMaxInputEl.addEventListener('change', drawDzSensPlot);

  bindEnterEscapeInput(dzPupilMaxInputEl, drawDzSensPlot);
  if (dzPupilMaxInputEl) dzPupilMaxInputEl.addEventListener('change', drawDzSensPlot);

const dzVmodeSelectEl = document.getElementById('dz-vmode-select');
if (dzVmodeSelectEl) dzVmodeSelectEl.addEventListener('change', drawDzVmodePlot);

vis.addEventListener('mousemove', updateMouseCoords);
vis.addEventListener('mouseleave', clearMouseCoords);
vis.addEventListener('wheel', (event) => {
  event.preventDefault();
  const rawStep = -event.deltaY * WHEEL_ZOOM_SENSITIVITY;
  const zoomStep = Math.max(-WHEEL_ZOOM_MAX_STEP, Math.min(WHEEL_ZOOM_MAX_STEP, rawStep));
  zoomBy(zoomStep);
}, {passive: false});

vis.addEventListener('mousedown', (event) => {
  if (event.button !== 0) return;
  isDraggingView = true;
  dragLastClientX = event.clientX;
  dragLastClientY = event.clientY;
  vis.style.cursor = 'grabbing';
});

vis.addEventListener('mousemove', (event) => {
  if (!isDraggingView) return;
  const dx = event.clientX - dragLastClientX;
  const dy = event.clientY - dragLastClientY;
  dragLastClientX = event.clientX;
  dragLastClientY = event.clientY;
  panByPixels(dx, dy);
});

window.addEventListener('mouseup', () => {
  if (!isDraggingView) return;
  isDraggingView = false;
  vis.style.cursor = '';
});

vis.addEventListener('mouseleave', () => {
  if (!isDraggingView) return;
  isDraggingView = false;
  vis.style.cursor = '';
});

if (zoomInBtn) {
  zoomInBtn.addEventListener('click', () => zoomBy(0.3));
}

if (zoomOutBtn) {
  zoomOutBtn.addEventListener('click', () => zoomBy(-0.3));
}

if (zoomResetBtn) {
  zoomResetBtn.addEventListener('click', resetView);
}

window.addEventListener('keydown', (event) => {
  if (event.defaultPrevented || event.metaKey || event.ctrlKey || event.altKey) return;
  const tag = event.target && event.target.tagName;
  if (tag === 'TEXTAREA' || tag === 'SELECT' || tag === 'INPUT') return;
  if (event.key === 'r' || event.key === 'R') {
    event.preventDefault();
    resetControls();
    return;
  }
  if (event.key === '+' || event.key === '=') {
    event.preventDefault();
    zoomBy(0.3);
    return;
  }
  if (event.key === '-') {
    event.preventDefault();
    zoomBy(-0.3);
    return;
  }
  if (event.key === '0') {
    event.preventDefault();
    resetView();
  }
});
