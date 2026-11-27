const MARK = { physics: "circle", biology: "square", project: "triangle", conversation: "diamond" };
const DURATION = 18000;
const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

const plot = document.querySelector("#plot");
const bands = document.querySelector("#bands");
const protocol = document.querySelector("#protocol");
const readout = document.querySelector("#readout");
const legend = document.querySelector("#legend");
const bead = document.querySelector("#bead");
const pauseButton = document.querySelector("#pause");

let geometry = null;
let paused = reduceMotion;
let pauseStarted = 0;
let elapsedAtPause = 0;
let origin = performance.now();
let holdCritic = false;
let phaseName = "";
let marks = [];
let bandDots = [];
let bandWindow = null;
let link = null;
let ghost = null;

function clamp(value, lo, hi) {
  return Math.min(hi, Math.max(lo, value));
}
function lerp(a, b, t) {
  return a + (b - a) * t;
}
function ease(t) {
  const x = clamp(t, 0, 1);
  return x * x * (3 - 2 * x);
}

function clock() {
  if (paused) return elapsedAtPause;
  const t = (performance.now() - origin) % DURATION;
  return reduceMotion ? 5200 : t;
}

function point(fragment) {
  const pad = 46;
  return [
    pad + ((fragment.x + 1) / 2) * (640 - pad * 2),
    36 + ((1 - fragment.y) / 2) * (320 - 28),
  ];
}

function byId(id) {
  return geometry.fragments.find((fragment) => fragment.id === id);
}

function phase(t) {
  if (t < 2200) return "enter";
  if (t < 6800) return "semantic_bridge";
  if (t < 11200) return "analogical_transfer";
  if (t < 14000) return "dialectic_synthesis";
  return "critic";
}

function local(t, start, end) {
  return clamp((t - start) / (end - start), 0, 1);
}

function addMark(fragment) {
  const node = document.createElementNS("http://www.w3.org/2000/svg", "g");
  const kind = MARK[fragment.domain] || "circle";
  const s = 5;
  let shape;
  if (kind === "square") {
    shape = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    shape.setAttribute("x", -s);
    shape.setAttribute("y", -s);
    shape.setAttribute("width", s * 2);
    shape.setAttribute("height", s * 2);
  } else if (kind === "triangle") {
    shape = document.createElementNS("http://www.w3.org/2000/svg", "polygon");
    shape.setAttribute("points", `0,${-s - 1} ${s + 1},${s} ${-s - 1},${s}`);
  } else if (kind === "diamond") {
    shape = document.createElementNS("http://www.w3.org/2000/svg", "polygon");
    shape.setAttribute("points", `0,${-s - 1} ${s + 1},0 0,${s + 1} ${-s - 1},0`);
  } else {
    shape = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    shape.setAttribute("r", s);
  }
  shape.setAttribute("fill", "#000");
  const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
  label.setAttribute("x", 8);
  label.setAttribute("y", -7);
  label.setAttribute("font-family", "Latin Modern Roman, serif");
  label.setAttribute("font-size", "13");
  label.textContent = String(fragment.id);
  node.append(shape, label);
  node.setAttribute("opacity", "0");
  plot.querySelector("#marks").append(node);
  return { fragment, node };
}

function buildScene() {
  const grid = plot.querySelector("#grid");
  grid.innerHTML = `<line x1="46" y1="188" x2="594" y2="188" stroke="#000" stroke-width="0.6"/>
    <line x1="320" y1="28" x2="320" y2="348" stroke="#000" stroke-width="0.6"/>`;
  const links = plot.querySelector("#links");
  ghost = document.createElementNS("http://www.w3.org/2000/svg", "line");
  ghost.setAttribute("stroke", "#000");
  ghost.setAttribute("stroke-width", "0.6");
  ghost.setAttribute("opacity", "0");
  link = document.createElementNS("http://www.w3.org/2000/svg", "line");
  link.setAttribute("stroke", "#000");
  link.setAttribute("stroke-width", "1.15");
  link.setAttribute("opacity", "0");
  links.append(ghost, link);
  marks = geometry.fragments.map(addMark);
  legend.textContent = "circle physics    square biology    triangle project    diamond conversation";

  const width = 640;
  const height = 150;
  const left = 28;
  const right = 612;
  const axisY = 96;
  bands.innerHTML = "";
  bandWindow = document.createElementNS("http://www.w3.org/2000/svg", "rect");
  bandWindow.setAttribute("y", "28");
  bandWindow.setAttribute("height", "58");
  bandWindow.setAttribute("fill", "#000");
  bandWindow.setAttribute("opacity", "0.08");
  bands.append(bandWindow);
  const axis = document.createElementNS("http://www.w3.org/2000/svg", "line");
  axis.setAttribute("x1", left);
  axis.setAttribute("x2", right);
  axis.setAttribute("y1", axisY);
  axis.setAttribute("y2", axisY);
  axis.setAttribute("stroke", "#000");
  bands.append(axis);
  for (const tick of [-0.2, 0, 0.2, 0.4, 0.6, 0.8, 1]) {
    const x = scaleCos(tick, left, right);
    const mark = document.createElementNS("http://www.w3.org/2000/svg", "line");
    mark.setAttribute("x1", x);
    mark.setAttribute("x2", x);
    mark.setAttribute("y1", axisY);
    mark.setAttribute("y2", axisY + 6);
    mark.setAttribute("stroke", "#000");
    bands.append(mark);
    const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
    label.setAttribute("x", x);
    label.setAttribute("y", "124");
    label.setAttribute("text-anchor", "middle");
    label.setAttribute("font-family", "Latin Modern Mono, ui-monospace, monospace");
    label.setAttribute("font-size", "11");
    label.textContent = tick.toFixed(1);
    bands.append(label);
  }
  bandDots = geometry.pairs.map((pair) => {
    const dot = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    dot.setAttribute("cy", "78");
    dot.setAttribute("r", "3");
    dot.setAttribute("fill", "#000");
    bands.append(dot);
    return { pair, dot, left, right };
  });
}

function scaleCos(value, left, right) {
  const min = -0.4;
  const max = 1;
  return left + ((value - min) / (max - min)) * (right - left);
}

function sentenceFor(name, pair) {
  if (name === "enter") return "Place the seven fragments of the public hash.";
  if (name === "dialectic_synthesis") {
    return "Dialectic synthesis. No pair is both lexically opposed and inside [0.20, 0.75].";
  }
  if (name === "critic") {
    return "Template critic. N = L = F = U = 0.50, so C = 0.500. Reject.";
  }
  if (!pair) return "";
  if (name === "semantic_bridge") {
    return `Semantic bridge. Pair (${pair.a}, ${pair.b}), cosine ${pair.cosine.toFixed(4)}, midpoint score ${pair.score.toFixed(4)}. Kept.`;
  }
  return `Analogical transfer. Pair (${pair.a}, ${pair.b}), cosine ${pair.cosine.toFixed(4)}. The score is the cosine. Kept.`;
}

function setProtocol(name, pair, t, start) {
  const full = sentenceFor(name, pair);
  if (phaseName !== name) {
    phaseName = name;
    protocol.setAttribute("aria-label", full);
  }
  if (reduceMotion || paused && holdCritic) {
    protocol.textContent = full;
    return;
  }
  const count = Math.floor(local(t, start, start + 900) * full.length);
  protocol.textContent = full.slice(0, count);
}

function placeLink(line, a, b, progress) {
  const [x1, y1] = point(a);
  const [x2, y2] = point(b);
  const len = Math.hypot(x2 - x1, y2 - y1) || 1;
  line.setAttribute("x1", x1);
  line.setAttribute("y1", y1);
  line.setAttribute("x2", x2);
  line.setAttribute("y2", y2);
  line.setAttribute("stroke-dasharray", String(len));
  line.setAttribute("stroke-dashoffset", String(len * (1 - progress)));
  line.setAttribute("opacity", progress > 0 ? "1" : "0");
  return [x1, y1, x2, y2, len];
}

function render(t) {
  const name = phase(t);
  const appearEnd = 2200;
  marks.forEach((item, index) => {
    const born = ease(local(t, index * 180, index * 180 + 420));
    const show = t >= appearEnd ? 1 : born;
    const [x, y] = point(item.fragment);
    item.node.setAttribute("transform", `translate(${x} ${y}) scale(${0.2 + 0.8 * show})`);
    item.node.setAttribute("opacity", String(show));
  });

  const windows = {
    semantic_bridge: geometry.thresholds.bridge,
    analogical_transfer: geometry.thresholds.analogy,
    dialectic_synthesis: geometry.thresholds.dialectic,
  };
  let pair = null;
  let progress = 0;
  let beadU = 0;
  let ghostPair = null;
  if (name === "semantic_bridge" || name === "analogical_transfer") {
    const start = name === "semantic_bridge" ? 2200 : 6800;
    pair = geometry.selected[name][0];
    progress = ease(local(t, start + 250, start + 1400));
    const travel = local(t, start + 1400, start + 4200);
    beadU = progress < 1 ? progress : 0.5 + 0.5 * Math.sin(travel * Math.PI * 2);
  } else if (name === "dialectic_synthesis") {
    const flashes = geometry.pairs.filter((item) => item.cosine < 0.2 || item.cosine > 0.75).slice(0, 6);
    const step = Math.floor(local(t, 11200, 14000) * flashes.length);
    ghostPair = flashes[Math.min(step, flashes.length - 1)];
    progress = (local(t, 11200, 14000) * flashes.length) % 1;
  }

  ghost.setAttribute("opacity", "0");
  if (pair) {
    const [x1, y1, x2, y2] = placeLink(link, byId(pair.a), byId(pair.b), progress);
    bead.setAttribute("cx", lerp(x1, x2, beadU));
    bead.setAttribute("cy", lerp(y1, y2, beadU));
    bead.setAttribute("opacity", progress > 0.05 ? "1" : "0");
    const shown = pair.cosine * progress;
    const score = pair.score * progress;
    readout.textContent = name === "semantic_bridge"
      ? `cos = ${shown.toFixed(4)}    S(m) = ${score.toFixed(4)}`
      : `cos = ${shown.toFixed(4)}`;
  } else if (ghostPair) {
    placeLink(ghost, byId(ghostPair.a), byId(ghostPair.b), ease(progress));
    link.setAttribute("opacity", "0");
    bead.setAttribute("opacity", "0");
    readout.textContent = "no eligible pair";
  } else {
    link.setAttribute("opacity", "0");
    bead.setAttribute("opacity", "0");
    readout.textContent = name === "critic" ? "C = 0.500    reject" : "";
  }

  const starts = { enter: 0, semantic_bridge: 2200, analogical_transfer: 6800, dialectic_synthesis: 11200, critic: 14000 };
  setProtocol(name, pair, t, starts[name]);

  const criticT = name === "critic" ? ease(local(t, 14200, 16000)) : 0;
  if (!holdCritic) paintBars(criticT, criticT, criticT, criticT);
  document.querySelectorAll("#eq-c .term").forEach((term, index) => {
    term.classList.toggle("on", name === "critic" && criticT > (index + 1) / 5);
  });

  const active = name === "enter" ? "semantic_bridge" : name === "critic" ? "semantic_bridge" : name;
  const window = windows[active];
  const open = name === "enter" ? ease(local(t, 1600, 2200)) : 1;
  const x0 = scaleCos(window[0], 28, 612);
  const x1 = scaleCos(window[1], 28, 612);
  bandWindow.setAttribute("x", x0);
  bandWindow.setAttribute("width", (x1 - x0) * open);
  const kept = new Set((geometry.selected[active] || []).slice(0, 1).map((item) => `${item.a}-${item.b}`));
  bandDots.forEach((item) => {
    const target = scaleCos(item.pair.cosine, item.left, item.right);
    const from = scaleCos(0, item.left, item.right);
    const slide = ease(local(t, 400, 2000));
    const key = `${item.pair.a}-${item.pair.b}`;
    const on = kept.has(key) || kept.has(`${item.pair.b}-${item.pair.a}`);
    item.dot.setAttribute("cx", lerp(from, target, reduceMotion ? 1 : slide));
    item.dot.setAttribute("cy", on ? "62" : "78");
    item.dot.setAttribute("r", on ? "5.5" : "3");
  });
}

function paintBars(n, l, f, u) {
  const values = { n, l, f, u };
  for (const [key, value] of Object.entries(values)) {
    document.querySelector(`#fill-${key}`).style.width = `${value * 100}%`;
    document.querySelector(`#num-${key}`).textContent = value.toFixed(2);
  }
  if (!holdCritic && phase(clock()) === "critic") {
    setScores([0.5, 0.5, 0.5, 0.5], false);
  }
}

function weights() {
  return geometry ? geometry.thresholds.weights : { novelty: 0.3, logical_coherence: 0.3, falsifiability: 0.2, cross_domain_utility: 0.2 };
}
function gates() {
  return geometry ? geometry.thresholds : { accept: 0.6, axis_floor: 0.3 };
}
function readScores() {
  return {
    novelty: Number(document.querySelector("#novelty").value),
    logical_coherence: Number(document.querySelector("#coherence").value),
    falsifiability: Number(document.querySelector("#falsifiability").value),
    cross_domain_utility: Number(document.querySelector("#utility").value),
  };
}
function renderCritic() {
  const scores = readScores();
  const w = weights();
  const gate = gates();
  for (const [name, value] of Object.entries(scores)) {
    document.querySelector(`[data-out="${name}"]`).textContent = value.toFixed(2);
  }
  const composite = scores.novelty * w.novelty + scores.logical_coherence * w.logical_coherence
    + scores.falsifiability * w.falsifiability + scores.cross_domain_utility * w.cross_domain_utility;
  const weak = Object.entries(scores).find(([, value]) => value < gate.axis_floor);
  const accept = composite + 1e-9 >= gate.accept && !weak;
  const formula = `C = ${composite.toFixed(3)}`;
  document.querySelector("#verdict").textContent = accept
    ? `${formula}. Accept, because C >= ${gate.accept.toFixed(2)} and every axis is at least ${gate.axis_floor.toFixed(2)}.`
    : weak
      ? `${formula}. Reject, because ${weak[0].replaceAll("_", " ")} is below ${gate.axis_floor.toFixed(2)}.`
      : `${formula}. Reject, because C is below ${gate.accept.toFixed(2)}.`;
}
function setScores(values, hold) {
  document.querySelector("#novelty").value = values[0];
  document.querySelector("#coherence").value = values[1];
  document.querySelector("#falsifiability").value = values[2];
  document.querySelector("#utility").value = values[3];
  if (hold) holdCritic = true;
  renderCritic();
}

function tick() {
  if (geometry) render(clock());
  if (!paused) requestAnimationFrame(tick);
}

pauseButton.addEventListener("click", () => {
  if (reduceMotion) return;
  if (!paused) {
    elapsedAtPause = clock();
    paused = true;
    pauseButton.textContent = "Play";
  } else {
    paused = false;
    origin = performance.now() - elapsedAtPause;
    pauseButton.textContent = "Pause";
    requestAnimationFrame(tick);
  }
});
if (reduceMotion) pauseButton.textContent = "Play";

document.querySelectorAll("#critic input").forEach((input) => {
  input.addEventListener("input", () => {
    holdCritic = true;
    if (!paused && !reduceMotion) pauseButton.click();
    renderCritic();
    const scores = readScores();
    paintBars(scores.novelty, scores.logical_coherence, scores.falsifiability, scores.cross_domain_utility);
  });
});
document.querySelector("#preset-template").addEventListener("click", () => setScores([0.5, 0.5, 0.5, 0.5], true));
document.querySelector("#preset-pass").addEventListener("click", () => setScores([0.8, 0.75, 0.7, 0.65], true));
document.querySelector("#preset-floor").addEventListener("click", () => setScores([0.9, 0.9, 0.2, 0.9], true));
document.querySelector("#copy").addEventListener("click", async () => {
  const command = document.querySelector("#command").textContent.trim();
  try {
    await navigator.clipboard.writeText(command);
    document.querySelector("#copy").textContent = "Copied";
  } catch {
    document.querySelector("#copy").textContent = "Select the line";
  }
});

renderCritic();

fetch("geometry.json")
  .then((response) => {
    if (!response.ok) throw new Error(String(response.status));
    return response.json();
  })
  .then((payload) => {
    geometry = payload;
    buildScene();
    renderCritic();
    requestAnimationFrame(tick);
  })
  .catch(() => {
    protocol.textContent = "The geometry file did not load. Table 1 is still the public corpus.";
  });
