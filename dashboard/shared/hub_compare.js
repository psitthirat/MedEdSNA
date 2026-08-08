// Cross-field comparison section on the hub page. Self-contained (like
// share.js/tooltip.js) -- reads window.HUB_COMPARISON (written by
// scripts/export_hub_comparison.py) rather than a single field's
// window.DASHBOARD_DATA, and reuses the same CSS tokens + tooltip.js
// infrastructure every field dashboard already uses, so a new field added
// to the hub picks up styling for free.
(function () {
  "use strict";

  const COMPARISON = window.HUB_COMPARISON;
  if (!COMPARISON || !COMPARISON.fields || !COMPARISON.fields.length) return;

  const css = getComputedStyle(document.documentElement);
  const tok = (name) => css.getPropertyValue(name).trim();

  // Field identity color, assigned by position -- the same blue/red pair
  // already used for every other two-series comparison on the field pages
  // (e.g. "Avg. clustering & modularity", "Homophily income & region").
  // Only 2 fields exist today. A 3rd needs a 3rd hue picked and validated
  // deliberately (see the dataviz skill) and added here -- never cycled or
  // borrowed from the reserved status palette.
  const FIELD_COLOR_VARS = ["--seq-450", "--div-neg"];
  const fieldColor = (i) => tok(FIELD_COLOR_VARS[i] || "--text-muted");

  const DIV_POS = tok("--div-pos");
  const DIV_NEG = tok("--div-neg");
  const DIV_MID = tok("--div-mid");
  const BASELINE = tok("--baseline");
  const SURFACE = tok("--surface-1");
  const INCOME_COLOR = {
    "High income": tok("--income-high"),
    "Upper middle income": tok("--income-upper-mid"),
    "Lower middle income": tok("--income-lower-mid"),
    "Low income": tok("--income-low"),
  };
  const fmtPct = (v) => (v * 100).toFixed(0) + "%";
  const fmt3 = (v) => (v === null || v === undefined ? "—" : v.toFixed(3));

  const fields = COMPARISON.fields;

  // ---------------------------------------------------------------------
  // Collaboration equity: one stacked bar per field, latest year available.
  // ---------------------------------------------------------------------
  function renderEquity() {
    const container = document.getElementById("compare-equity");
    if (!container) return;
    container.innerHTML = "";

    fields.forEach((field) => {
      const series = field.collaboration_equity_series;
      if (!series || !series.length) return;
      const latest = series[series.length - 1];

      const row = document.createElement("div");
      row.className = "equity-row";
      row.innerHTML = `<span class="equity-field-label">${field.name}</span>`;

      const bar = document.createElement("div");
      bar.className = "equity-bar";
      [
        { key: "high_only_share", label: "High-income only", color: DIV_POS },
        { key: "developing_only_share", label: "Developing-country only", color: DIV_NEG },
        { key: "mixed_share", label: "Mixed", color: DIV_MID },
      ].forEach((seg) => {
        const share = latest[seg.key] || 0;
        if (share <= 0) return;
        const el = document.createElement("span");
        el.style.width = fmtPct(share);
        el.style.background = seg.color;
        el.dataset.label = seg.label;
        el.dataset.share = share;
        bar.appendChild(el);
      });
      bar.addEventListener("mousemove", (evt) => {
        const html = `<div class="tt-title">${field.name}, ${latest.year}</div>` +
          [
            { label: "High-income only", key: "high_only_share", color: DIV_POS },
            { label: "Developing-country only", key: "developing_only_share", color: DIV_NEG },
            { label: "Mixed", key: "mixed_share", color: DIV_MID },
          ].map((seg) =>
            `<div class="tt-row"><span style="color:${seg.color}">● ${seg.label}</span><span>${fmtPct(latest[seg.key] || 0)}</span></div>`
          ).join("");
        showTooltip(html, evt);
      });
      bar.addEventListener("mouseleave", hideTooltip);

      row.appendChild(bar);
      container.appendChild(row);
    });

    const legendEl = document.getElementById("compare-equity-legend");
    if (legendEl) {
      legendEl.innerHTML = [
        { label: "High-income only", color: DIV_POS },
        { label: "Developing-country only", color: DIV_NEG },
        { label: "Mixed group", color: DIV_MID },
      ].map((l) => `<span class="legend-item"><span class="swatch" style="background:${l.color}"></span>${l.label}</span>`).join("");
    }
  }

  // ---------------------------------------------------------------------
  // Small-world coefficient: one line per field, over its full year range.
  // ---------------------------------------------------------------------
  function renderSmallWorld() {
    const svgId = "chart-compare-smallworld";
    const el = document.getElementById(svgId);
    if (!el) return;
    const width = el.clientWidth || 500;
    const height = +el.getAttribute("height");
    // Extra right margin for a direct value-label at each line's end (mark
    // spec: "lines -> value at the end"), so identity+value both read
    // without leaning on the legend alone.
    const margin = { top: 10, right: 34, bottom: 20, left: 32 };
    const innerW = width - margin.left - margin.right;
    const innerH = height - margin.top - margin.bottom;

    const s = d3.select("#" + svgId).attr("viewBox", `0 0 ${width} ${height}`);
    s.selectAll("*").remove();
    const g = s.append("g").attr("transform", `translate(${margin.left},${margin.top})`);

    const allRows = fields.flatMap((f) => f.small_world_series);
    const years = allRows.map((r) => r.year);
    const x = d3.scaleLinear().domain(d3.extent(years)).range([8, innerW]);
    const y = d3.scaleLinear().domain([0, d3.max(allRows, (r) => r.value) * 1.1 || 1]).nice().range([innerH, 0]);

    g.append("g").selectAll("line").data(y.ticks(4)).join("line")
      .attr("class", "gridline").attr("x1", 0).attr("x2", innerW).attr("y1", y).attr("y2", y);
    g.append("g").attr("class", "axis").attr("transform", `translate(0,${innerH})`)
      .call(d3.axisBottom(x).tickFormat(d3.format("d")).ticks(Math.min(6, d3.extent(years)[1] - d3.extent(years)[0])).tickSize(0))
      .call((sel) => sel.select(".domain").attr("stroke", BASELINE));
    g.append("g").attr("class", "axis")
      .call(d3.axisLeft(y).ticks(4).tickSize(0).tickFormat((d) => d.toFixed(1)))
      .call((sel) => sel.select(".domain").remove());

    const lineFor = d3.line().x((d) => x(d.year)).y((d) => y(d.value)).curve(d3.curveMonotoneX);
    // Soft wash under each line -- a background layer of color rather than
    // a flat card, per the mark spec's area-fill rule (series hue, ~10%).
    const areaFor = d3.area().x((d) => x(d.year)).y0(innerH).y1((d) => y(d.value)).curve(d3.curveMonotoneX);

    fields.forEach((field, i) => {
      const rows = field.small_world_series;
      if (!rows.length) return;
      const color = fieldColor(i);
      g.append("path").attr("fill", color).attr("fill-opacity", 0.08).attr("stroke", "none").attr("d", areaFor(rows));
      g.append("path").attr("fill", "none").attr("stroke", color).attr("stroke-width", 2).attr("d", lineFor(rows));
      const last = rows[rows.length - 1];
      g.append("circle").attr("cx", x(last.year)).attr("cy", y(last.value)).attr("r", 3.5)
        .attr("fill", color).attr("stroke", SURFACE).attr("stroke-width", 1.5);
      g.append("text").attr("class", "direct-label muted")
        .attr("x", x(last.year) + 7).attr("y", y(last.value) + 3.5)
        .style("font-size", "10px").text(last.value.toFixed(1));
    });

    const legendEl = document.getElementById("compare-smallworld-legend");
    if (legendEl) {
      legendEl.innerHTML = fields.map((f, i) =>
        `<span class="legend-item"><span class="swatch" style="background:${fieldColor(i)}"></span>${f.name}</span>`
      ).join("");
    }

    // crosshair + tooltip, one row per field at the hovered year
    const focusLine = g.append("line").attr("class", "gridline").attr("y1", 0).attr("y2", innerH).style("opacity", 0);
    s.append("rect")
      .attr("x", margin.left).attr("y", margin.top).attr("width", innerW).attr("height", innerH)
      .attr("fill", "transparent")
      .on("mousemove", (evt) => {
        const [mx] = d3.pointer(evt, g.node());
        const yr = Math.round(x.invert(mx));
        const present = fields.filter((f) => f.small_world_series.some((r) => r.year === yr));
        if (!present.length) return;
        focusLine.attr("x1", x(yr)).attr("x2", x(yr)).style("opacity", 1);
        const html = `<div class="tt-title">${yr}</div>` + fields.map((f, i) => {
          const row = f.small_world_series.find((r) => r.year === yr);
          if (!row) return "";
          return `<div class="tt-row"><span style="color:${fieldColor(i)}">● ${f.name}</span><span>${row.value.toFixed(2)}</span></div>`;
        }).join("");
        showTooltip(html, evt);
      })
      .on("mouseleave", () => { focusLine.style("opacity", 0); hideTooltip(); });
  }

  // ---------------------------------------------------------------------
  // Top-N most central countries per field (degree centrality, full network).
  // ---------------------------------------------------------------------
  function renderLeaders() {
    const container = document.getElementById("compare-leaders");
    if (!container) return;
    container.innerHTML = "";

    fields.forEach((field) => {
      const col = document.createElement("div");
      col.className = "leaders-col";
      const items = field.top_leaders.map((l, i) => `
        <li>
          <span class="rank">${i + 1}.</span>
          <span class="dot" style="background:${INCOME_COLOR[l.income_group] || "var(--text-muted)"}"></span>
          <span class="name">${l.economy}</span>
          <span class="value">${fmt3(l.degree_centrality)}</span>
        </li>`).join("");
      col.innerHTML = `<h4>${field.name}</h4><ul class="leaders-list">${items}</ul>`;
      container.appendChild(col);
    });
  }

  function init() {
    renderEquity();
    renderSmallWorld();
    renderLeaders();
    let resizeTimer;
    window.addEventListener("resize", () => {
      clearTimeout(resizeTimer);
      resizeTimer = setTimeout(renderSmallWorld, 150);
    });
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
