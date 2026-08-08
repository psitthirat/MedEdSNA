// Shared hover tooltip + info-icon (ⓘ) definitions -- used by app.js (map,
// network, and per-field charts) and by hub_compare.js (cross-field
// comparison charts on the hub page). Deliberately not wrapped in an IIFE:
// showTooltip/hideTooltip are meant to be called as bare globals from every
// chart-rendering script on the page, the same way each of those scripts
// itself declares its top-level functions.
//
// Load this before app.js / hub_compare.js, and make sure the page has a
// `<div id="tooltip"></div>` somewhere in the body.

const tooltipEl = document.getElementById("tooltip");

function showTooltip(html, evt) {
  tooltipEl.innerHTML = html;
  tooltipEl.style.opacity = 1;
  moveTooltip(evt);
}

function moveTooltip(evt) {
  const pad = 14;
  let x = evt.clientX + pad;
  let y = evt.clientY + pad;
  const rect = tooltipEl.getBoundingClientRect();
  if (x + rect.width > window.innerWidth) x = evt.clientX - rect.width - pad;
  if (y + rect.height > window.innerHeight) y = evt.clientY - rect.height - pad;
  tooltipEl.style.left = x + "px";
  tooltipEl.style.top = y + "px";
}

function hideTooltip() { tooltipEl.style.opacity = 0; }

// ---------------------------------------------------------------------
// Metric definitions (info-icon hover, wired via delegation so it covers
// both static markup and chart titles re-rendered by D3)
// ---------------------------------------------------------------------
const METRIC_DEFINITIONS = {
  "nodes-density": {
    label: "Nodes & density",
    text: "Nodes: number of countries active in the network that year. Density: the share of all possible country-pairs that are actually connected by at least one co-authored publication (0 = none connected, 1 = every pair connected).",
  },
  "small-world": {
    label: "Small-world coefficient",
    text: "Compares the network's clustering and path length to a random network of the same size. Values above 1 indicate small-world structure — tightly clustered locally, yet still reachable in a few steps globally; higher is more pronounced.",
  },
  "clustering-modularity": {
    label: "Avg. clustering & modularity",
    text: "Avg. clustering: how often a country's collaborators are also connected to each other. Modularity: how strongly the network splits into distinct clusters that collaborate mostly within themselves rather than across.",
  },
  homophily: {
    label: "Homophily",
    text: "The share of collaboration ties that connect two countries in the same income group or region, rather than crossing between groups. Higher means more within-group clustering and less cross-group mixing.",
  },
  degree: {
    label: "Degree centrality",
    text: "The share of all other countries in the network that a country is directly connected to. Higher means more direct collaborators.",
  },
  betweenness: {
    label: "Betweenness centrality",
    text: "How often a country sits on the shortest connecting path between two other countries. High values mark a 'broker' bridging otherwise separate parts of the network.",
  },
  closeness: {
    label: "Closeness centrality",
    text: "How close a country is, on average, to every other country via the shortest paths. Higher means faster reach across the whole network.",
  },
  participation: {
    label: "Participation coefficient",
    text: "How evenly a country's ties are spread across different communities rather than concentrated in its own. Near 1 = broadly connected across communities; near 0 = ties stay within one community.",
  },
  "collab-equity": {
    label: "Collaboration equity",
    text: "Each publication classified by the income groups of its author countries: High-income only, Developing-country only (no high-income co-author), or Mixed (both). Shares are of that field's total for the year shown.",
  },
};

document.addEventListener("mousemove", (evt) => {
  const icon = evt.target.closest(".info-icon");
  if (!icon) return;
  const def = METRIC_DEFINITIONS[icon.dataset.metric];
  if (!def) return;
  showTooltip(`<div class="tt-title">${def.label}</div><div class="tt-desc">${def.text}</div>`, evt);
});
document.addEventListener("mouseout", (evt) => {
  const icon = evt.target.closest(".info-icon");
  if (!icon) return;
  if (evt.relatedTarget && evt.relatedTarget.closest(".info-icon") === icon) return;
  hideTooltip();
});
