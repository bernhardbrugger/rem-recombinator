const MARK = { physics: "circle", biology: "square", project: "triangle", conversation: "diamond" };
const DURATION = 23000;
const PHASES = [
  { name: "enter", at: 0 },
  { name: "semantic_bridge", at: 2800 },
  { name: "analogical_transfer", at: 8600 },
  { name: "dialectic_synthesis", at: 14200 },
  { name: "critic", at: 17600 },
];
const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

const plot = document.querySelector("#plot");
const bands = document.querySelector("#bands");
const protocol = document.querySelector("#protocol");
const stage = document.querySelector("#stage");
const readout = document.querySelector("#readout");
const bead = document.querySelector("#bead");
const pauseButton = document.querySelector("#pause");

const FRAME = { l: 86, r: 624, t: 28, b: 378 };
const PAD = 0.22;
let geometry = null;
let paused = reduceMotion;
let elapsedAtPause = reduceMotion ? 7000 : 0;
let origin = performance.now();
let holdCritic = false;
let phaseName = "";
let hotTerms = -1;
let marks = [];
let bandDots = [];
let bandWindow = null;
let cursor = null;
let link = null;
let ghost = null;

const TEX = {
  cos: String.raw`\cos(u,v)=\dfrac{u\cdot v}{\lVert u\rVert\,\lVert v\rVert}.`,
  mid: String.raw`\begin{aligned} m &= \operatorname{normalize}\!\bigl(\operatorname{normalize}(u)+\operatorname{normalize}(v)\bigr),\\ S &= 1-\max_{k\notin\{i,j\}}\cos(m,v_k). \end{aligned}`,
  gate: String.raw`\begin{aligned} & \text{accept if } C\ge 0.60\\ & \text{and } \min(N,L,F,U)\ge 0.30. \end{aligned}`,
};

function typeset(el, tex) {
  if (!el || !window.katex) return;
  window.katex.render(tex, el, { displayMode: true, throwOnError: false, output: "html" });
}

function criticTex(hot) {
  const terms = [["0.30", "N"], ["0.30", "L"], ["0.20", "F"], ["0.20", "U"]];
  const body = terms.map(([weight, symbol], index) => {
    const atom = `${weight}\\,${symbol}`;
    return index < hot ? `\\underline{${atom}}` : atom;
  }).join("+");
  return `C = ${body}.`;
}

function clamp(value, lo, hi) { return Math.min(hi, Math.max(lo, value)); }
function lerp(a, b, t) { return a + (b - a) * t; }
function ease(t) {
  const x = clamp(t, 0, 1);
  return x * x * (3 - 2 * x);
}
function local(t, start, end) { return clamp((t - start) / (end - start), 0, 1); }

function clock() {
  if (paused) return elapsedAtPause;
  return (performance.now() - origin) % DURATION;
}

function phaseAt(t) {
  let current = PHASES[0];
  for (const item of PHASES) if (t >= item.at) current = item;
  return current;
}

function xy(fragment) {
  const span = 2 + 2 * PAD;
  return [
    FRAME.l + ((fragment.x + 1 + PAD) / span) * (FRAME.r - FRAME.l),
    FRAME.b - ((fragment.y + 1 + PAD) / span) * (FRAME.b - FRAME.t),
  ];
}

function byId(id) {
  return geometry.fragments.find((fragment) => fragment.id === id);
}

function scaleCos(value, left, right) {
  const min = -0.4;
  const max = 1;
  return left + ((value - min) / (max - min)) * (right - left);
}

function ns(name) {
  return document.createElementNS("http://www.w3.org/2000/svg", name);
}

function addMark(fragment) {
  const node = ns("g");
  const kind = MARK[fragment.domain] || "circle";
  const s = 5.2;
  let shape;
  if (kind === "square") {
    shape = ns("rect");
    shape.setAttribute("x", -s);
    shape.setAttribute("y", -s);
    shape.setAttribute("width", s * 2);
    shape.setAttribute("height", s * 2);
  } else if (kind === "triangle") {
    shape = ns("polygon");
    shape.setAttribute("points", `0,${-s - 1} ${s + 1.2},${s} ${-s - 1.2},${s}`);
  } else if (kind === "diamond") {
    shape = ns("polygon");
    shape.setAttribute("points", `0,${-s - 1} ${s + 1},0 0,${s + 1} ${-s - 1},0`);
  } else {
    shape = ns("circle");
    shape.setAttribute("r", s);
  }
  shape.setAttribute("fill", "#000");
  const label = ns("text");
  label.setAttribute("x", 8);
  label.setAttribute("y", -8);
  label.setAttribute("font-family", "Latin Modern Roman, serif");
  label.setAttribute("font-size", "13");
  label.textContent = String(fragment.id);
  node.append(shape, label);
  node.setAttribute("opacity", "0");
  plot.querySelector("#marks").append(node);
  return { fragment, node };
}

function buildScene() {
  const frame = plot.querySelector("#frame");
  const box = ns("rect");
  box.setAttribute("x", FRAME.l);
  box.setAttribute("y", FRAME.t);
  box.setAttribute("width", FRAME.r - FRAME.l);
  box.setAttribute("height", FRAME.b - FRAME.t);
  box.setAttribute("fill", "none");
  box.setAttribute("stroke", "#000");
  box.setAttribute("stroke-width", "0.9");
  frame.append(box);
  const pc1 = ((geometry.explained_variance[0] || 0) * 100).toFixed(1);
  const pc2 = ((geometry.explained_variance[1] || 0) * 100).toFixed(1);
  for (const tick of [-1, -0.5, 0, 0.5, 1]) {
    const [x] = xy({ x: tick, y: -1 });
    const [, y] = xy({ x: -1, y: tick });
    const xt = ns("line");
    xt.setAttribute("x1", x);
    xt.setAttribute("x2", x);
    xt.setAttribute("y1", FRAME.b);
    xt.setAttribute("y2", FRAME.b + 5);
    xt.setAttribute("stroke", "#000");
    xt.setAttribute("stroke-width", "0.7");
    const xl = ns("text");
    xl.setAttribute("x", x);
    xl.setAttribute("y", FRAME.b + 18);
    xl.setAttribute("text-anchor", "middle");
    xl.setAttribute("font-family", "Latin Modern Roman, serif");
    xl.setAttribute("font-size", "15");
    xl.textContent = tick.toFixed(1);
    const yt = ns("line");
    yt.setAttribute("x1", FRAME.l - 5);
    yt.setAttribute("x2", FRAME.l);
    yt.setAttribute("y1", y);
    yt.setAttribute("y2", y);
    yt.setAttribute("stroke", "#000");
    yt.setAttribute("stroke-width", "0.7");
    const yl = ns("text");
    yl.setAttribute("x", FRAME.l - 8);
    yl.setAttribute("y", y + 4);
    yl.setAttribute("text-anchor", "end");
    yl.setAttribute("font-family", "Latin Modern Roman, serif");
    yl.setAttribute("font-size", "15");
    yl.textContent = tick.toFixed(1);
    frame.append(xt, xl, yt, yl);
  }
  const xlab = ns("text");
  xlab.setAttribute("x", (FRAME.l + FRAME.r) / 2);
  xlab.setAttribute("y", 432);
  xlab.setAttribute("text-anchor", "middle");
  xlab.setAttribute("font-family", "Latin Modern Roman, serif");
  xlab.setAttribute("font-size", "16");
  xlab.textContent = `PC1, ${pc1}% of the hash variance`;
  const ylab = ns("text");
  ylab.setAttribute("x", 18);
  ylab.setAttribute("y", (FRAME.t + FRAME.b) / 2);
  ylab.setAttribute("text-anchor", "middle");
  ylab.setAttribute("transform", `rotate(-90 18 ${(FRAME.t + FRAME.b) / 2})`);
  ylab.setAttribute("font-family", "Latin Modern Roman, serif");
  ylab.setAttribute("font-size", "16");
  ylab.textContent = `PC2, ${pc2}%`;
  frame.append(xlab, ylab);

  ghost = ns("line");
  ghost.setAttribute("stroke", "#000");
  ghost.setAttribute("stroke-width", "0.7");
  ghost.setAttribute("opacity", "0");
  link = ns("line");
  link.setAttribute("stroke", "#000");
  link.setAttribute("stroke-width", "1.35");
  link.setAttribute("opacity", "0");
  plot.querySelector("#links").append(ghost, link);
  marks = geometry.fragments.map(addMark);

  const left = 48;
  const right = 624;
  const axisY = 104;
  bands.innerHTML = "";
  bandWindow = ns("rect");
  bandWindow.setAttribute("y", "46");
  bandWindow.setAttribute("height", "58");
  bandWindow.setAttribute("fill", "#000");
  bandWindow.setAttribute("opacity", "0.08");
  bands.append(bandWindow);
  const axis = ns("line");
  axis.setAttribute("x1", left);
  axis.setAttribute("x2", right);
  axis.setAttribute("y1", axisY);
  axis.setAttribute("y2", axisY);
  axis.setAttribute("stroke", "#000");
  bands.append(axis);
  cursor = ns("line");
  cursor.setAttribute("y1", "42");
  cursor.setAttribute("y2", "112");
  cursor.setAttribute("stroke", "#000");
  cursor.setAttribute("stroke-width", "0.9");
  cursor.setAttribute("opacity", "0");
  bands.append(cursor);
  for (const tick of [-0.2, 0, 0.2, 0.4, 0.6, 0.8, 1]) {
    const x = scaleCos(tick, left, right);
    const mark = ns("line");
    mark.setAttribute("x1", x);
    mark.setAttribute("x2", x);
    mark.setAttribute("y1", axisY);
    mark.setAttribute("y2", axisY + 6);
    mark.setAttribute("stroke", "#000");
    bands.append(mark);
    const label = ns("text");
    label.setAttribute("x", x);
    label.setAttribute("y", "132");
    label.setAttribute("text-anchor", "middle");
    label.setAttribute("font-family", "Latin Modern Mono, ui-monospace, monospace");
    label.setAttribute("font-size", "12");
    label.textContent = tick.toFixed(1);
    bands.append(label);
  }
  const cosLabel = ns("text");
  cosLabel.setAttribute("x", (left + right) / 2);
  cosLabel.setAttribute("y", "22");
  cosLabel.setAttribute("text-anchor", "middle");
  cosLabel.setAttribute("font-family", "Latin Modern Roman, serif");
  cosLabel.setAttribute("font-style", "italic");
  cosLabel.setAttribute("font-size", "14");
  cosLabel.textContent = "cosine of the pair";
  bands.append(cosLabel);
  bandDots = geometry.pairs.map((pair) => {
    const dot = ns("circle");
    dot.setAttribute("cy", "86");
    dot.setAttribute("r", "3.1");
    dot.setAttribute("fill", "#000");
    bands.append(dot);
    return { pair, dot, left, right };
  });
}

function sentence(name, pair) {
  if (name === "enter") return "Seven fragments of the public hash, on the first two principal components.";
  if (name === "semantic_bridge" && pair) {
    return `Semantic bridge keeps (${pair.a}, ${pair.b}). The fragments sit in different clusters, the cosine ${pair.cosine.toFixed(4)} lies in [0.05, 0.35], and the midpoint score is ${pair.score.toFixed(4)}.`;
  }
  if (name === "analogical_transfer" && pair) {
    return `Analogical transfer keeps (${pair.a}, ${pair.b}). The domains differ, the cosine ${pair.cosine.toFixed(4)} lies in [0.15, 0.55], and that cosine is the score.`;
  }
  if (name === "dialectic_synthesis") {
    return "Dialectic synthesis keeps nothing. No pair is both lexically opposed and inside [0.20, 0.75].";
  }
  return "The template scores every axis at 0.50. Then C = 0.500, below 0.60, so the report is rejected.";
}

function stageLabel(name) {
  if (name === "semantic_bridge") return "semantic bridge";
  if (name === "analogical_transfer") return "analogical transfer";
  if (name === "dialectic_synthesis") return "dialectic synthesis";
  if (name === "critic") return "template critic";
  return "";
}

function bandFor(name) {
  if (name === "analogical_transfer") return geometry.thresholds.analogy;
  if (name === "dialectic_synthesis") return geometry.thresholds.dialectic;
  return geometry.thresholds.bridge;
}

function placeLink(line, a, b, progress) {
  const [x1, y1] = xy(a);
  const [x2, y2] = xy(b);
  const len = Math.hypot(x2 - x1, y2 - y1) || 1;
  line.setAttribute("x1", x1);
  line.setAttribute("y1", y1);
  line.setAttribute("x2", x2);
  line.setAttribute("y2", y2);
  line.setAttribute("stroke-dasharray", String(len));
  line.setAttribute("stroke-dashoffset", String(len * (1 - progress)));
  line.setAttribute("opacity", progress > 0 ? "1" : "0");
  return [x1, y1, x2, y2];
}

function render(t) {
  const phase = phaseAt(t);
  const name = phase.name;
  marks.forEach((item, index) => {
    const born = ease(local(t, 500 + index * 160, 980 + index * 160));
    const show = t >= 2800 ? 1 : born;
    const [x, y] = xy(item.fragment);
    item.node.setAttribute("transform", `translate(${x} ${y}) scale(${0.25 + 0.75 * show})`);
    item.node.setAttribute("opacity", String(show));
  });

  let pair = null;
  let progress = 0;
  let beadU = 0;
  let ghostPair = null;
  if (name === "semantic_bridge" || name === "analogical_transfer") {
    pair = geometry.selected[name][0];
    progress = ease(local(t, phase.at + 200, phase.at + 1500));
    const settle = ease(local(t, phase.at + 1500, phase.at + 2300));
    beadU = progress < 1 ? progress : lerp(1, 0.5, settle);
  } else if (name === "dialectic_synthesis") {
    const flashes = geometry.pairs.filter((item) => item.cosine < 0.2 || item.cosine > 0.75).slice(0, 5);
    const span = local(t, phase.at, phase.at + 2800) * flashes.length;
    ghostPair = flashes[Math.min(Math.floor(span), flashes.length - 1)];
    progress = ease(span % 1);
  }

  ghost.setAttribute("opacity", "0");
  if (pair) {
    const [x1, y1, x2, y2] = placeLink(link, byId(pair.a), byId(pair.b), progress);
    bead.setAttribute("cx", lerp(x1, x2, beadU));
    bead.setAttribute("cy", lerp(y1, y2, beadU));
    const pulse = progress < 1 ? 1 : 0.7 + 0.3 * Math.sin((t - phase.at) / 260);
    bead.setAttribute("opacity", progress > 0.04 ? String(pulse) : "0");
    readout.textContent = name === "semantic_bridge"
      ? `cos = ${(pair.cosine * progress).toFixed(4)}    S = ${(pair.score * progress).toFixed(4)}`
      : `cos = ${(pair.cosine * progress).toFixed(4)}`;
  } else if (ghostPair) {
    placeLink(ghost, byId(ghostPair.a), byId(ghostPair.b), progress);
    link.setAttribute("opacity", "0");
    bead.setAttribute("opacity", "0");
    readout.textContent = "no eligible pair";
  } else {
    link.setAttribute("opacity", name === "critic" ? "0" : link.getAttribute("opacity"));
    if (name !== "semantic_bridge" && name !== "analogical_transfer") {
      link.setAttribute("opacity", "0");
      bead.setAttribute("opacity", "0");
    }
    readout.textContent = name === "critic" ? "C = 0.500    reject" : "";
  }

  const full = sentence(name, pair);
  if (phaseName !== name) phaseName = name;
  stage.textContent = stageLabel(name);
  stage.setAttribute("opacity", String(ease(local(t, phase.at, phase.at + 420))));
  protocol.style.opacity = String(ease(local(t, phase.at, phase.at + 420)));
  protocol.textContent = full;

  const criticT = name === "critic" ? ease(local(t, phase.at + 200, phase.at + 1800)) : 0;
  const hot = name === "critic" ? Math.floor(criticT * 4) : 0;
  if (hot !== hotTerms) {
    hotTerms = hot;
    typeset(document.querySelector("#tex-3"), criticTex(hot));
  }
  const level = 0.5 * criticT;
  if (!holdCritic) paintBars(level, level, level, level);

  const shown = name === "critic" ? "dialectic_synthesis" : name === "enter" ? "semantic_bridge" : name;
  const interval = bandFor(shown);
  const open = name === "enter" ? ease(local(t, 1600, 2600)) : ease(local(t, phase.at, phase.at + 700));
  const x0 = scaleCos(interval[0], 48, 624);
  const x1 = scaleCos(interval[1], 48, 624);
  bandWindow.setAttribute("x", x0);
  bandWindow.setAttribute("width", Math.max(0, (x1 - x0) * (name === "enter" ? open : 1)));
  const kept = new Set((geometry.selected[shown] || []).slice(0, 1).map((item) => `${item.a}-${item.b}`));
  let cursorX = null;
  bandDots.forEach((item) => {
    const target = scaleCos(item.pair.cosine, item.left, item.right);
    const from = scaleCos(0, item.left, item.right);
    const slide = ease(local(t, 600, 2200));
    const key = `${item.pair.a}-${item.pair.b}`;
    const on = kept.has(key) || kept.has(`${item.pair.b}-${item.pair.a}`);
    item.dot.setAttribute("cx", lerp(from, target, reduceMotion ? 1 : slide));
    item.dot.setAttribute("cy", on ? "68" : "86");
    item.dot.setAttribute("r", on ? "5.4" : "3.1");
    if (on) cursorX = target;
  });
  if (cursorX == null || name === "dialectic_synthesis" || name === "enter") {
    cursor.setAttribute("opacity", "0");
  } else {
    const from = scaleCos(interval[0], 48, 624);
    cursor.setAttribute("x1", lerp(from, cursorX, open));
    cursor.setAttribute("x2", lerp(from, cursorX, open));
    cursor.setAttribute("opacity", String(0.9 * open));
  }
}

function paintBars(n, l, f, u) {
  const values = { n, l, f, u };
  for (const [key, value] of Object.entries(values)) {
    document.querySelector(`#fill-${key}`).style.width = `${value * 100}%`;
    document.querySelector(`#num-${key}`).textContent = value.toFixed(2);
  }
  if (!holdCritic && phaseAt(clock()).name === "critic") setScores([0.5, 0.5, 0.5, 0.5], false);
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
    ? `${formula}. Accept, because C is at least ${gate.accept.toFixed(2)} and every axis is at least ${gate.axis_floor.toFixed(2)}.`
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

typeset(document.querySelector("#tex-1"), TEX.cos);
typeset(document.querySelector("#tex-2"), TEX.mid);
typeset(document.querySelector("#tex-3"), criticTex(0));
typeset(document.querySelector("#tex-4"), TEX.gate);
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
