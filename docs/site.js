const COLORS = {
  physics: "#0f6e56",
  biology: "#8c3b1e",
  project: "#1e4d7b",
  conversation: "#8a6a2f",
};

const STRATEGIES = [
  ["semantic_bridge", "Semantic bridge"],
  ["analogical_transfer", "Analogical transfer"],
  ["dialectic_synthesis", "Dialectic synthesis"],
];

const FALLBACK = {
  accept: 0.6,
  axis_floor: 0.3,
  weights: { novelty: 0.3, logical_coherence: 0.3, falsifiability: 0.2, cross_domain_utility: 0.2 },
};

let geometry = null;
let strategy = "semantic_bridge";
let pinned = null;

const plot = document.querySelector("#plot");
const pairList = document.querySelector("#pairs");
const caption = document.querySelector("#plot-caption");
const bands = document.querySelector("#bands");
const strategyButtons = document.querySelector("#strategies");

function point(fragment) {
  const pad = 46;
  const width = 640;
  const height = 460;
  return [
    pad + ((fragment.x + 1) / 2) * (width - pad * 2),
    pad + ((1 - fragment.y) / 2) * (height - pad * 2),
  ];
}

function byId(id) {
  return geometry.fragments.find((fragment) => fragment.id === id);
}

function renderStrategies() {
  strategyButtons.innerHTML = "";
  for (const [key, label] of STRATEGIES) {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = label;
    button.setAttribute("aria-pressed", String(key === strategy));
    button.addEventListener("click", () => {
      strategy = key;
      pinned = null;
      renderStrategies();
      renderPlot();
      renderBands();
    });
    strategyButtons.appendChild(button);
  }
}

function renderPlot() {
  const selected = geometry.selected[strategy] || [];
  const active = new Set(selected.flatMap((pair) => [pair.a, pair.b]));
  const parts = [
    `<rect width="640" height="460" fill="#fbf8f1"></rect>`,
    `<line x1="46" y1="230" x2="594" y2="230" stroke="#d5ccbc"></line>`,
    `<line x1="320" y1="46" x2="320" y2="414" stroke="#d5ccbc"></line>`,
  ];
  for (const pair of selected) {
    const left = point(byId(pair.a));
    const right = point(byId(pair.b));
    parts.push(
      `<line x1="${left[0]}" y1="${left[1]}" x2="${right[0]}" y2="${right[1]}" stroke="#1c1914" stroke-width="1.4"></line>`
    );
  }
  for (const fragment of geometry.fragments) {
    const [x, y] = point(fragment);
    const color = COLORS[fragment.domain] || "#1c1914";
    const emphasized = active.has(fragment.id);
    const radius = emphasized ? 8 : 5.5;
    const mark = pinned === fragment.id ? ` stroke="#1c1914" stroke-width="2"` : "";
    parts.push(
      `<circle class="dot" data-id="${fragment.id}" cx="${x}" cy="${y}" r="${radius}" fill="${color}"${mark}></circle>`
    );
    parts.push(
      `<text x="${x + 12}" y="${y + 4}" font-family="IBM Plex Mono, ui-monospace, monospace" font-size="11" fill="#1c1914" stroke="#fbf8f1" stroke-width="3" paint-order="stroke">${fragment.id}</text>`
    );
  }
  plot.innerHTML = parts.join("");
  plot.querySelectorAll(".dot").forEach((dot) => {
    dot.style.cursor = "pointer";
    dot.addEventListener("click", () => {
      pinned = Number(dot.getAttribute("data-id"));
      renderPlot();
      renderPairs();
    });
  });
  const variance = geometry.explained_variance;
  const shown = variance.map((value) => `${(value * 100).toFixed(1)}%`).join(" and ");
  caption.textContent =
    `Figure 1. ${geometry.disclaimer} The plane keeps ${shown} of the hash variance. ` +
    `Lines are the highest-scoring eligible pairs for the selected rule, at most three. ` +
    (selected.length
      ? `${selected.length} pair${selected.length === 1 ? "" : "s"} qualify.`
      : "No pair meets this rule.");
  renderPairs();
}

function renderPairs() {
  const selected = geometry.selected[strategy] || [];
  const pinnedFragment = pinned ? byId(pinned) : null;
  const head = pinnedFragment
    ? `<h3>${pinnedFragment.id}. ${pinnedFragment.domain}</h3><p>${pinnedFragment.text}</p>`
    : `<h3>${labelFor(strategy)}</h3><p>Click a point. The number is the fragment id in the public table.</p>`;
  const rows = selected
    .map((pair) => {
      const left = byId(pair.a);
      const right = byId(pair.b);
      return `<div class="pair"><b>${pair.a} × ${pair.b}</b> · cosine ${pair.cosine.toFixed(4)} · score ${pair.score.toFixed(4)}<br>${left.domain}: ${left.text}<br>${right.domain}: ${right.text}</div>`;
    })
    .join("");
  pairList.innerHTML = head + (rows || `<div class="pair">The lexical opposition list and the cosine window do not fire together on this sample.</div>`);
}

function labelFor(key) {
  return STRATEGIES.find((item) => item[0] === key)[1];
}

function renderBands() {
  const thresholds = geometry.thresholds;
  const windows = {
    semantic_bridge: thresholds.bridge,
    analogical_transfer: thresholds.analogy,
    dialectic_synthesis: thresholds.dialectic,
  };
  const min = -0.4;
  const max = 1;
  const width = 640;
  const height = 168;
  const left = 16;
  const right = 624;
  const scale = (value) => left + ((value - min) / (max - min)) * (right - left);
  const window = windows[strategy];
  const selectedKeys = new Set((geometry.selected[strategy] || []).map((pair) => `${pair.a}-${pair.b}`));
  const parts = [`<rect width="${width}" height="${height}" fill="#fbf8f1"></rect>`];
  parts.push(
    `<rect x="${scale(window[0])}" y="28" width="${scale(window[1]) - scale(window[0])}" height="78" fill="#d7ebe4"></rect>`
  );
  parts.push(`<line x1="${left}" y1="108" x2="${right}" y2="108" stroke="#1c1914"></line>`);
  for (const tick of [-0.2, 0, 0.2, 0.4, 0.6, 0.8, 1]) {
    const x = scale(tick);
    parts.push(`<line x1="${x}" y1="108" x2="${x}" y2="114" stroke="#1c1914"></line>`);
    parts.push(
      `<text x="${x}" y="132" text-anchor="middle" font-family="IBM Plex Mono, ui-monospace, monospace" font-size="11" fill="#5c564c">${tick.toFixed(1)}</text>`
    );
  }
  for (const pair of geometry.pairs) {
    const x = scale(pair.cosine);
    const on = selectedKeys.has(`${pair.a}-${pair.b}`) || selectedKeys.has(`${pair.b}-${pair.a}`);
    parts.push(
      `<circle cx="${x}" cy="${on ? 62 : 86}" r="${on ? 5 : 3}" fill="${on ? "#1c1914" : "#8a8175"}"></circle>`
    );
  }
  parts.push(
    `<text x="16" y="22" font-family="IBM Plex Mono, ui-monospace, monospace" font-size="12" fill="#1c1914">${labelFor(strategy)} window [${window[0].toFixed(2)}, ${window[1].toFixed(2)}]</text>`
  );
  bands.innerHTML = parts.join("");
}

function weights() {
  return geometry ? geometry.thresholds.weights : FALLBACK.weights;
}

function gates() {
  if (!geometry) return FALLBACK;
  return geometry.thresholds;
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
  const composite =
    scores.novelty * w.novelty +
    scores.logical_coherence * w.logical_coherence +
    scores.falsifiability * w.falsifiability +
    scores.cross_domain_utility * w.cross_domain_utility;
  const floorName = Object.entries(scores).find(([, value]) => value < gate.axis_floor);
  const accept = composite >= gate.accept && !floorName;
  const verdict = document.querySelector("#verdict");
  verdict.className = `verdict ${accept ? "accept" : "reject"}`;
  const formula =
    `C = ${w.novelty.toFixed(2)}·${scores.novelty.toFixed(2)} + ${w.logical_coherence.toFixed(2)}·${scores.logical_coherence.toFixed(2)} + ${w.falsifiability.toFixed(2)}·${scores.falsifiability.toFixed(2)} + ${w.cross_domain_utility.toFixed(2)}·${scores.cross_domain_utility.toFixed(2)} = ${composite.toFixed(3)}`;
  const because = accept
    ? `Accept. C ≥ ${gate.accept.toFixed(2)} and every axis is at least ${gate.axis_floor.toFixed(2)}.`
    : floorName
      ? `Reject. ${floorName[0].replaceAll("_", " ")} is below the axis floor ${gate.axis_floor.toFixed(2)}, even when the composite clears ${gate.accept.toFixed(2)}.`
      : `Reject. C is below ${gate.accept.toFixed(2)}.`;
  verdict.textContent = `${formula}. ${because}`;
}

function setScores(values) {
  document.querySelector("#novelty").value = values[0];
  document.querySelector("#coherence").value = values[1];
  document.querySelector("#falsifiability").value = values[2];
  document.querySelector("#utility").value = values[3];
  renderCritic();
}

document.querySelectorAll("#critic input").forEach((input) => {
  input.addEventListener("input", renderCritic);
});
document.querySelector("#preset-template").addEventListener("click", () => setScores([0.5, 0.5, 0.5, 0.5]));
document.querySelector("#preset-pass").addEventListener("click", () => setScores([0.8, 0.75, 0.7, 0.65]));
document.querySelector("#preset-floor").addEventListener("click", () => setScores([0.9, 0.9, 0.2, 0.9]));

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
    renderStrategies();
    renderPlot();
    renderBands();
    renderCritic();
    const bridge = payload.selected.semantic_bridge[0];
    const analogy = payload.selected.analogical_transfer[0];
    const note = document.querySelector("#reading");
    note.textContent =
      `On this hash, the top semantic bridge is fragments ${bridge.a} and ${bridge.b} ` +
      `(cosine ${bridge.cosine.toFixed(4)}, midpoint score ${bridge.score.toFixed(4)}). ` +
      `The top analogy is fragments ${analogy.a} and ${analogy.b} (cosine ${analogy.cosine.toFixed(4)}). ` +
      `Dialectic synthesis selects nothing: no pair is both lexically opposed and inside its cosine window. ` +
      `These are properties of the offline hash, not of a semantic model.`;
  })
  .catch(() => {
    caption.textContent = "The geometry file did not load. The table below is still the public corpus.";
  });
