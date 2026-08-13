(function () {
  "use strict";

  const DATA = window.DASHBOARD_DATA;
  const WORLD = window.WORLD_GEOJSON;

  // ---------------------------------------------------------------------
  // Design tokens (mirrors style.css — kept in sync manually)
  // ---------------------------------------------------------------------
  const css = getComputedStyle(document.documentElement);
  const tok = (name) => css.getPropertyValue(name).trim();

  const COUNTRY_NAME = new Map(DATA.country_stats.map((c) => [c.ISO_A2_EH, c.Economy]));
  const nameFor = (iso) => COUNTRY_NAME.get(iso) || iso;

  const INCOME_ORDER = ["High income", "Upper middle income", "Lower middle income", "Low income"];

  // Color tokens are re-read from CSS whenever the theme toggle fires (see
  // refreshColorTokens/wireThemeToggle) -- `let`, not `const`, so every
  // render function that closes over them picks up the new values.
  let INCOME_COLOR, SEQ_ACCENT, SEQ_RAMP, DIV_POS, DIV_NEG, DIV_MID, TEXT_MUTED, GRIDLINE, BASELINE, SURFACE;
  function refreshColorTokens() {
    INCOME_COLOR = {
      "High income": tok("--income-high"),
      "Upper middle income": tok("--income-upper-mid"),
      "Lower middle income": tok("--income-lower-mid"),
      "Low income": tok("--income-low"),
    };
    SEQ_ACCENT = tok("--seq-450");
    SEQ_RAMP = d3.interpolateRgbBasis([
      tok("--seq-100"), tok("--seq-250"), tok("--seq-350"), SEQ_ACCENT, tok("--seq-550"), tok("--seq-700"),
    ]);
    DIV_POS = tok("--div-pos");
    DIV_NEG = tok("--div-neg");
    DIV_MID = tok("--div-mid");
    TEXT_MUTED = tok("--text-muted");
    GRIDLINE = tok("--gridline");
    BASELINE = tok("--baseline");
    SURFACE = tok("--surface-1");
  }
  refreshColorTokens();

  // ---------------------------------------------------------------------
  // Shared state
  // ---------------------------------------------------------------------
  const state = {
    mapMode: "choropleth", // 'choropleth' | 'network'
    networkVariant: "network_full",
    yearMin: DATA.summary.year_min,
    yearMax: DATA.summary.year_max,
    activeIncome: new Set(INCOME_ORDER),
  };

  const fmtCompact = (n) => {
    if (n === null || n === undefined || Number.isNaN(n)) return "—";
    const abs = Math.abs(n);
    if (abs >= 1e6) return (n / 1e6).toFixed(1) + "M";
    if (abs >= 1e3) return (n / 1e3).toFixed(1) + "K";
    return d3.format(",")(Math.round(n));
  };
  const fmtDec = (n, d = 2) => (n === null || n === undefined || Number.isNaN(n) ? "—" : n.toFixed(d));

  // ---------------------------------------------------------------------
  // Tooltip + info-icon definitions now live in shared/tooltip.js (loaded
  // before this file) -- showTooltip/hideTooltip are its globals.
  // ---------------------------------------------------------------------

  // ---------------------------------------------------------------------
  // KPI row (hero)
  // ---------------------------------------------------------------------
  const MONTHS_SHORT = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
  const fmtMonthYear = (iso) => {
    const d = new Date(iso + "T00:00:00");
    return `${MONTHS_SHORT[d.getMonth()]} ${d.getFullYear()}`;
  };

  function renderKPIRow() {
    const s = DATA.summary;
    const tiles = [
      { label: "Publications", value: fmtCompact(s.total_publications) },
      { label: "Contributing authors", value: fmtCompact(s.total_authors) },
      { label: "Countries represented", value: fmtCompact(s.total_countries) },
      { label: "Institutions", value: fmtCompact(s.total_institutions) },
      { label: "Journals", value: fmtCompact(s.total_journals), tooltip: s.journal_names },
      { label: "Coverage", value: `${fmtMonthYear(s.date_min)} – ${fmtMonthYear(s.date_max)}` },
    ];
    const row = d3.select("#kpi-row").selectAll(".stat-tile").data(tiles).join("div").attr("class", "stat-tile");
    row.classed("has-tooltip", (d) => !!d.tooltip);
    row.html((d) => `<p class="label">${d.label}</p><p class="value">${d.value}</p>`);
    row
      .on("mousemove", (evt, d) => {
        if (!d.tooltip) return;
        const lines = (Array.isArray(d.tooltip) ? d.tooltip : [d.tooltip])
          .map((line) => `<div class="tt-desc">${line}</div>`)
          .join("");
        showTooltip(`<div class="tt-title">${d.label}</div>${lines}`, evt);
      })
      .on("mouseleave", (evt, d) => { if (d.tooltip) hideTooltip(); });
  }

  // ---------------------------------------------------------------------
  // World map: projection + base landmasses
  // ---------------------------------------------------------------------
  const svg = d3.select("#world-svg");
  const gZoom = svg.append("g").attr("id", "zoom-root");
  const gLand = gZoom.append("g").attr("class", "layer-land");
  const gChoropleth = gZoom.append("g").attr("class", "layer-choropleth");
  const gEdges = gZoom.append("g").attr("class", "layer-edges");
  const gNodes = gZoom.append("g").attr("class", "layer-nodes");

  const zoomBehavior = d3.zoom()
    .scaleExtent([1, 8])
    .filter((event) => {
      if (event.type === "wheel") return event.ctrlKey || event.metaKey;
      if (event.type === "mousedown") return !event.button;
      return true;
    })
    .on("zoom", (event) => {
      gZoom.attr("transform", event.transform);
    });

  let projection = d3.geoNaturalEarth1();
  let path = d3.geoPath(projection);
  let centroids = new Map(); // ISO_A2_EH -> [x, y]
  let featureByIso = new Map();

  // Manual node-position overrides [lon, lat], for countries whose polygon
  // centroid (e.g. pulled by overseas territories, or just an odd shape)
  // doesn't land where the node should visually read as "based."
  const MANUAL_NODE_POSITIONS = {
    FR: [2.2137, 46.2276], // mainland France
    AU: [133.3917, -22.1583], // center of Australia's bounding box (0°41'S-43°38'S, 113°09'E-153°38'E)
  };

  (WORLD.features || []).forEach((f) => featureByIso.set(f.properties.ISO_A2_EH, f));

  function fitProjection() {
    const el = document.getElementById("map-layer");
    const w = el.clientWidth;
    const h = el.clientHeight;
    svg.attr("viewBox", `0 0 ${w} ${h}`);
    projection.fitSize([w * 1.08, h * 1.08], WORLD);
    // slight upward bias so landmasses sit behind content rather than centered on it
    const [tx, ty] = projection.translate();
    projection.translate([tx, ty - h * 0.03]);
    path = d3.geoPath(projection);

    centroids = new Map();
    featureByIso.forEach((f, iso) => {
      const c = path.centroid(f);
      if (!Number.isNaN(c[0])) centroids.set(iso, c);
    });
    Object.entries(MANUAL_NODE_POSITIONS).forEach(([iso, lonLat]) => {
      const p = projection(lonLat);
      if (p) centroids.set(iso, p);
    });

    gLand.selectAll("path").attr("d", path);

    zoomBehavior
      .translateExtent([[-w * 0.5, -h * 0.5], [w * 1.5, h * 1.5]])
      .extent([[0, 0], [w, h]]);
    svg.call(zoomBehavior.transform, d3.zoomIdentity);

    renderMapLayer();
  }

  function drawBaseLand() {
    gLand.selectAll("path")
      .data(WORLD.features)
      .join("path")
      .attr("class", "map-land")
      .attr("d", path);
  }

  // ---------------------------------------------------------------------
  // Choropleth
  // ---------------------------------------------------------------------
  function aggregateCountryPubs(yearMin, yearMax) {
    const totals = new Map();
    DATA.country_year_stats.forEach((r) => {
      if (r.year >= yearMin && r.year <= yearMax) {
        totals.set(r.ISO_A2_EH, (totals.get(r.ISO_A2_EH) || 0) + r.publications);
      }
    });
    return totals;
  }

  function renderChoropleth() {
    gEdges.selectAll("*").remove();
    gNodes.selectAll("*").remove();

    const totals = aggregateCountryPubs(state.yearMin, state.yearMax);
    const meta = new Map(DATA.country_stats.map((c) => [c.ISO_A2_EH, c]));
    const maxLog = d3.max(Array.from(totals.values()), (v) => Math.log1p(v)) || 1;
    // Red-yellow-green, matching the notebook's RdYlGn choropleth: red = low
    // publication counts, green = high.
    const colorScale = d3.scaleSequential(d3.interpolateRdYlGn).domain([0, maxLog]);

    const paths = gChoropleth.selectAll("path").data(WORLD.features, (d) => d.properties.ISO_A2_EH);
    paths.join("path")
      .attr("class", "country-fill")
      .attr("d", path)
      .attr("fill", (d) => {
        const v = totals.get(d.properties.ISO_A2_EH);
        return v ? colorScale(Math.log1p(v)) : "transparent";
      })
      .classed("is-dimmed", (d) => {
        const m = meta.get(d.properties.ISO_A2_EH);
        return m ? !state.activeIncome.has(m["Income group"]) : false;
      })
      .style("pointer-events", "auto")
      .style("cursor", (d) => (totals.get(d.properties.ISO_A2_EH) ? "pointer" : "default"))
      .on("mousemove", (evt, d) => {
        const iso = d.properties.ISO_A2_EH;
        const v = totals.get(iso);
        if (!v) return hideTooltip();
        const m = meta.get(iso) || {};
        showTooltip(
          `<div class="tt-title">${m.Economy || d.properties.ADMIN}</div>
           <div class="tt-row"><span>Publications (${state.yearMin}–${state.yearMax})</span><span>${fmtCompact(v)}</span></div>
           <div class="tt-row"><span>Income group</span><span>${m["Income group"] || "—"}</span></div>`,
          evt
        );
      })
      .on("mouseleave", hideTooltip);

    const maxPub = Math.expm1(maxLog);
    const legendStops = d3.range(0, 1.0001, 0.1).map((t) => colorScale(t * maxLog)).join(", ");
    document.getElementById("map-legend").innerHTML = `
      <p class="ml-title">Publications, ${state.yearMin}–${state.yearMax}</p>
      <div class="ml-bar" style="background:linear-gradient(90deg, ${legendStops})"></div>
      <div class="ml-scale"><span>1</span><span>${fmtCompact(Math.round(maxPub))}</span></div>
    `;
  }

  // ---------------------------------------------------------------------
  // Network on map
  // ---------------------------------------------------------------------
  // Competition ranking (ties share the better rank; matches the pandas
  // .rank(method="min") convention used elsewhere in this dashboard).
  function rankMap(nodes, key) {
    const valid = nodes.filter((n) => n[key] !== null && n[key] !== undefined && !Number.isNaN(n[key]));
    const sorted = valid.slice().sort((a, b) => b[key] - a[key]);
    const map = new Map();
    let rank = 0, prevVal = null, seen = 0;
    sorted.forEach((n) => {
      seen++;
      if (n[key] !== prevVal) { rank = seen; prevVal = n[key]; }
      map.set(n.id, rank);
    });
    map.total = valid.length;
    return map;
  }
  function fmtRank(map, id) {
    const r = map.get(id);
    return r ? ` (#${r})` : "";
  }

  function renderNetwork() {
    // Network mode has its own node tooltip; suppress the choropleth's
    // country-area tooltip/pointer so hovering the landmass under a node
    // doesn't fight with it.
    gChoropleth.selectAll("path")
      .attr("fill", "transparent")
      .classed("is-dimmed", false)
      .style("pointer-events", "none")
      .on("mousemove", null)
      .on("mouseleave", null);

    const net = DATA[state.networkVariant];
    const nodeById = new Map(net.nodes.map((n) => [n.id, n]));
    const sizeExtent = d3.extent(net.nodes, (n) => n.size);
    const rScale = d3.scaleSqrt().domain([sizeExtent[0] || 1, sizeExtent[1] || 1]).range([2.5, 20]);
    const wExtent = d3.extent(net.edges, (e) => e.weight);
    const wScale = d3.scaleLinear().domain([wExtent[0] || 1, wExtent[1] || 1]).range([0.4, 4]);

    const degRank = rankMap(net.nodes, "degree_centrality");
    const btwRank = rankMap(net.nodes, "betweenness_centrality");
    const clsRank = rankMap(net.nodes, "closeness_centrality");
    const pcRank = rankMap(net.nodes, "participation_coefficient");

    // Strongest collaborator per node (by co-authored-publication edge
    // weight), one pass over the edge list rather than per-hover.
    const topCollabByNode = new Map();
    net.edges.forEach((e) => {
      [[e.source, e.target], [e.target, e.source]].forEach(([from, to]) => {
        const cur = topCollabByNode.get(from);
        if (!cur || e.weight > cur.weight) topCollabByNode.set(from, { id: to, weight: e.weight });
      });
    });

    const edges = gEdges.selectAll("path").data(net.edges, (d) => d.source + "-" + d.target);
    edges.join("path")
      .attr("class", "network-edge")
      .attr("d", (d) => {
        const a = centroids.get(d.source), b = centroids.get(d.target);
        if (!a || !b) return null;
        const mx = (a[0] + b[0]) / 2, my = (a[1] + b[1]) / 2 - Math.hypot(b[0] - a[0], b[1] - a[1]) * 0.06;
        return `M${a[0]},${a[1]} Q${mx},${my} ${b[0]},${b[1]}`;
      })
      .attr("stroke-width", (d) => wScale(d.weight))
      .attr("opacity", 0.28)
      .classed("is-dimmed", (d) => {
        const a = nodeById.get(d.source), b = nodeById.get(d.target);
        return !(a && b && state.activeIncome.has(a.income_group) && state.activeIncome.has(b.income_group));
      });

    const nodes = gNodes.selectAll("circle").data(net.nodes, (d) => d.id);
    nodes.join("circle")
      .attr("class", "network-node")
      .attr("cx", (d) => (centroids.get(d.id) || [null])[0])
      .attr("cy", (d) => (centroids.get(d.id) || [null, null])[1])
      .attr("r", (d) => rScale(d.size))
      .attr("fill", (d) => INCOME_COLOR[d.income_group] || TEXT_MUTED)
      .classed("is-dimmed", (d) => !state.activeIncome.has(d.income_group))
      .style("display", (d) => (centroids.get(d.id) ? null : "none"))
      .on("mousemove", (evt, d) => {
        const top = topCollabByNode.get(d.id);
        const topRow = top
          ? `<div class="tt-row"><span>Top collaborator</span><span>${nameFor(top.id)} (${fmtCompact(top.weight)})</span></div>`
          : "";
        showTooltip(
          `<div class="tt-title">${nameFor(d.id)}</div>
           <div class="tt-row"><span>Publications</span><span>${fmtCompact(d.size)}</span></div>
           <div class="tt-row"><span>Income group</span><span>${d.income_group || "—"}</span></div>
           <div class="tt-row"><span>Degree centrality</span><span>${fmtDec(d.degree_centrality, 3)}${fmtRank(degRank, d.id)}</span></div>
           <div class="tt-row"><span>Betweenness</span><span>${fmtDec(d.betweenness_centrality, 3)}${fmtRank(btwRank, d.id)}</span></div>
           <div class="tt-row"><span>Closeness</span><span>${fmtDec(d.closeness_centrality, 3)}${fmtRank(clsRank, d.id)}</span></div>
           <div class="tt-row"><span>Participation coeff.</span><span>${fmtDec(d.participation_coefficient, 3)}${fmtRank(pcRank, d.id)}</span></div>
           ${topRow}`,
          evt
        );
      })
      .on("mouseleave", hideTooltip);

    document.getElementById("map-legend").innerHTML = `
      <p class="ml-title">Income group (node color)</p>
      <div class="ml-dots">
        ${INCOME_ORDER.map((g) => `<div class="ml-dot-row"><span class="ml-dot" style="background:${INCOME_COLOR[g]}"></span>${g}</div>`).join("")}
      </div>
      <p class="ml-title" style="margin-top:10px;">Node size = publications</p>
    `;
  }

  function renderMapLayer() {
    if (state.mapMode === "choropleth") renderChoropleth();
    else renderNetwork();
  }

  // ---------------------------------------------------------------------
  // Trend line charts (publications / authors by income group)
  // ---------------------------------------------------------------------
  function lineChart(svgSel, records, valueKey, yLabel) {
    const el = document.getElementById(svgSel.replace("#", ""));
    const width = el.clientWidth || 480;
    const height = +el.getAttribute("height");
    const margin = { top: 16, right: 92, bottom: 24, left: 40 };
    const innerW = width - margin.left - margin.right;
    const innerH = height - margin.top - margin.bottom;

    const s = d3.select(svgSel).attr("viewBox", `0 0 ${width} ${height}`);
    s.selectAll("*").remove();
    const g = s.append("g").attr("transform", `translate(${margin.left},${margin.top})`);

    const years = Array.from(new Set(records.map((r) => r.year))).sort((a, b) => a - b);
    const x = d3.scaleLinear().domain(d3.extent(years)).range([8, innerW]);
    const y = d3.scaleLinear().domain([0, d3.max(records, (r) => r[valueKey]) * 1.08]).nice().range([innerH, 0]);

    // gridlines
    g.append("g").selectAll("line").data(y.ticks(4)).join("line")
      .attr("class", "gridline").attr("x1", 0).attr("x2", innerW).attr("y1", y).attr("y2", y);
    g.append("g").attr("class", "axis").attr("transform", `translate(0,${innerH})`)
      .call(d3.axisBottom(x).tickValues(years.filter((yr) => yr % 2 === 1 && yr !== years[0])).tickFormat(d3.format("d")).tickSize(0))
      .call((sel) => sel.select(".domain").attr("stroke", BASELINE));
    g.append("g").attr("class", "axis")
      .call(d3.axisLeft(y).ticks(4).tickFormat((d) => fmtCompact(d)).tickSize(0))
      .call((sel) => sel.select(".domain").remove());

    const byGroup = d3.group(records, (r) => r["Income group"]);
    const line = d3.line().x((d) => x(d.year)).y((d) => y(d[valueKey])).curve(d3.curveMonotoneX);

    const seriesG = g.append("g");
    const labelPositions = [];

    INCOME_ORDER.forEach((grp) => {
      const rows = (byGroup.get(grp) || []).slice().sort((a, b) => a.year - b.year);
      if (!rows.length) return;
      const active = state.activeIncome.has(grp);
      const color = INCOME_COLOR[grp];

      seriesG.append("path")
        .attr("fill", "none")
        .attr("stroke", color)
        .attr("stroke-width", 2)
        .attr("opacity", active ? 1 : 0.12)
        .attr("d", line(rows));

      const last = rows[rows.length - 1];
      seriesG.append("circle")
        .attr("cx", x(last.year)).attr("cy", y(last[valueKey])).attr("r", 4)
        .attr("fill", color).attr("stroke", SURFACE).attr("stroke-width", 2)
        .attr("opacity", active ? 1 : 0.12);

      if (active) labelPositions.push({ grp, y: y(last[valueKey]), color });
    });

    // simple label collision resolution (min 13px gap), sorted by y
    labelPositions.sort((a, b) => a.y - b.y);
    for (let i = 1; i < labelPositions.length; i++) {
      if (labelPositions[i].y - labelPositions[i - 1].y < 13) labelPositions[i].y = labelPositions[i - 1].y + 13;
    }
    seriesG.selectAll(".line-label").data(labelPositions).join("text")
      .attr("class", "direct-label")
      .attr("x", innerW + 8).attr("y", (d) => d.y + 3.5)
      .attr("fill", (d) => d.color)
      .style("font-size", "10.5px")
      .text((d) => d.grp.replace(" income", ""));

    g.append("text").attr("class", "axis-label").attr("x", -margin.left + 4).attr("y", -4).text(yLabel);

    // crosshair + tooltip
    const focusLine = g.append("line").attr("class", "gridline").attr("y1", 0).attr("y2", innerH).style("opacity", 0);
    s.append("rect")
      .attr("x", margin.left).attr("y", margin.top).attr("width", innerW).attr("height", innerH)
      .attr("fill", "transparent")
      .on("mousemove", (evt) => {
        const [mx] = d3.pointer(evt, g.node());
        const yr = Math.round(x.invert(mx));
        if (yr < years[0] || yr > years[years.length - 1]) return;
        focusLine.attr("x1", x(yr)).attr("x2", x(yr)).style("opacity", 1);
        const rows = records
          .filter((r) => r.year === yr && state.activeIncome.has(r["Income group"]))
          .sort((a, b) => INCOME_ORDER.indexOf(a["Income group"]) - INCOME_ORDER.indexOf(b["Income group"]));
        const html = `<div class="tt-title">${yr}</div>` + rows.map((r) =>
          `<div class="tt-row"><span style="color:${INCOME_COLOR[r["Income group"]]}">● ${r["Income group"]}</span><span>${fmtCompact(r[valueKey])}</span></div>`
        ).join("");
        showTooltip(html, evt);
      })
      .on("mouseleave", () => { focusLine.style("opacity", 0); hideTooltip(); });
  }

  function renderTrendCharts() {
    lineChart("#chart-trend-pubs", DATA.yearly_trends.publications, "publications", "Publications");
    lineChart("#chart-trend-authors", DATA.yearly_trends.authors, "authors", "Authors");
  }

  // ---------------------------------------------------------------------
  // Equity diverging chart
  // ---------------------------------------------------------------------
  function renderEquityChart() {
    const el = document.getElementById("chart-equity");
    const width = el.clientWidth || 900;
    const height = +el.getAttribute("height");
    const margin = { top: 16, right: 16, bottom: 24, left: 48 };
    const innerW = width - margin.left - margin.right;
    const innerH = height - margin.top - margin.bottom;

    const s = d3.select("#chart-equity").attr("viewBox", `0 0 ${width} ${height}`);
    s.selectAll("*").remove();
    const g = s.append("g").attr("transform", `translate(${margin.left},${margin.top})`);

    const rows = DATA.collaboration_equity.slice().sort((a, b) => a.year - b.year);
    const maxVal = d3.max(rows, (r) => Math.max(r["High only"] || 0, r["Developing only"] || 0));
    const x = d3.scaleBand().domain(rows.map((r) => r.year)).range([0, innerW]).padding(0.32);
    const y = d3.scaleLinear().domain([-maxVal * 1.08, maxVal * 1.08]).nice().range([innerH, 0]);
    const y0 = y(0);

    g.append("g").selectAll("line").data(y.ticks(5)).join("line")
      .attr("class", "gridline").attr("x1", 0).attr("x2", innerW).attr("y1", y).attr("y2", y);
    g.append("line").attr("class", "baseline").attr("x1", 0).attr("x2", innerW).attr("y1", y0).attr("y2", y0);

    g.append("g").attr("class", "axis").attr("transform", `translate(0,${innerH})`)
      .call(d3.axisBottom(x).tickValues(rows.map((r) => r.year).filter((yr) => yr % 2 === 1)).tickSize(0))
      .call((sel) => sel.select(".domain").remove());
    g.append("g").attr("class", "axis")
      .call(d3.axisLeft(y).ticks(5).tickFormat((d) => fmtCompact(Math.abs(d))).tickSize(0))
      .call((sel) => sel.select(".domain").remove());

    const bw = x.bandwidth();
    const mixScale = d3.scaleLinear().domain([0, d3.max(rows, (r) => r["Mixed"] || 0)]).range([0, Math.min(18, innerH * 0.18)]);

    const bar = g.selectAll(".eq-group").data(rows).join("g").attr("class", "eq-group")
      .attr("transform", (d) => `translate(${x(d.year)},0)`);

    bar.append("rect")
      .attr("x", 1).attr("width", bw - 2)
      .attr("y", (d) => y((d["High only"] || 0)))
      .attr("height", (d) => y0 - y(d["High only"] || 0))
      .attr("rx", 3)
      .attr("fill", DIV_POS);
    bar.append("rect")
      .attr("x", 1).attr("width", bw - 2)
      .attr("y", y0)
      .attr("height", (d) => y(-(d["Developing only"] || 0)) - y0)
      .attr("rx", 3)
      .attr("fill", DIV_NEG);
    bar.append("rect")
      .attr("x", bw / 2 - 3).attr("width", 6)
      .attr("y", (d) => y0 - mixScale(d["Mixed"] || 0))
      .attr("height", (d) => mixScale(d["Mixed"] || 0) * 2)
      .attr("fill", DIV_MID);

    bar.append("rect")
      .attr("x", 0).attr("width", bw).attr("y", 0).attr("height", innerH)
      .attr("fill", "transparent")
      .on("mousemove", (evt, d) => {
        showTooltip(
          `<div class="tt-title">${d.year}</div>
           <div class="tt-row"><span style="color:${DIV_POS}">● High-income only</span><span>${fmtCompact(d["High only"] || 0)}</span></div>
           <div class="tt-row"><span style="color:${DIV_NEG}">● Developing-country only</span><span>${fmtCompact(d["Developing only"] || 0)}</span></div>
           <div class="tt-row"><span style="color:${DIV_MID}">● Mixed group</span><span>${fmtCompact(d["Mixed"] || 0)}</span></div>`,
          evt
        );
      })
      .on("mouseleave", hideTooltip);

    d3.select("#equity-legend").html(`
      <div class="legend-item"><span class="swatch" style="background:${DIV_POS}"></span>High-income only</div>
      <div class="legend-item"><span class="swatch" style="background:${DIV_NEG}"></span>Developing-country only</div>
      <div class="legend-item"><span class="swatch" style="background:${DIV_MID}"></span>Mixed group</div>
    `);
  }

  // ---------------------------------------------------------------------
  // UpSet-style combination chart
  // ---------------------------------------------------------------------
  function renderUpsetChart() {
    const el = document.getElementById("chart-upset");
    const width = el.clientWidth || 900;
    const height = +el.getAttribute("height");
    const margin = { top: 8, right: 8, bottom: 8, left: 130 };
    const gap = 16;
    const rowH = 22;
    const rows = INCOME_ORDER;
    const barH = height - margin.top - margin.bottom - gap - rows.length * rowH;
    const innerW = width - margin.left - margin.right;

    const combos = DATA.upset_combinations.slice().sort((a, b) => b.count - a.count).slice(0, 10);

    const s = d3.select("#chart-upset").attr("viewBox", `0 0 ${width} ${height}`);
    s.selectAll("*").remove();
    const g = s.append("g").attr("transform", `translate(${margin.left},${margin.top})`);

    const x = d3.scaleBand().domain(combos.map((_, i) => i)).range([0, innerW]).padding(0.35);
    const y = d3.scaleLinear().domain([0, d3.max(combos, (d) => d.count)]).nice().range([barH, 0]);

    g.append("g").selectAll("line").data(y.ticks(4)).join("line")
      .attr("class", "gridline").attr("x1", 0).attr("x2", innerW).attr("y1", y).attr("y2", y);
    g.append("g").attr("class", "axis").call(d3.axisLeft(y).ticks(4).tickFormat(fmtCompact).tickSize(0))
      .call((sel) => sel.select(".domain").remove());

    const bars = g.selectAll(".bar").data(combos).join("g")
      .attr("transform", (d, i) => `translate(${x(i)},0)`);
    bars.append("rect")
      .attr("y", (d) => y(d.count)).attr("width", x.bandwidth()).attr("height", (d) => barH - y(d.count))
      .attr("rx", 4).attr("fill", SEQ_ACCENT);
    bars.append("text")
      .attr("class", "direct-label").attr("x", x.bandwidth() / 2).attr("y", (d) => y(d.count) - 6)
      .attr("text-anchor", "middle").style("font-size", "10px")
      .text((d) => fmtCompact(d.count));

    // dot matrix
    const matrixTop = barH + gap;
    g.append("g").selectAll(".row-label").data(rows).join("text")
      .attr("x", -10).attr("y", (d, i) => matrixTop + i * rowH + rowH / 2 + 3)
      .attr("text-anchor", "end").attr("class", "axis-label").style("font-size", "11px")
      .attr("fill", (d) => INCOME_COLOR[d])
      .text((d) => d);

    combos.forEach((combo, ci) => {
      const cx = x(ci) + x.bandwidth() / 2;
      const present = rows.map((r) => combo[r]);
      g.append("line")
        .attr("class", "gridline")
        .attr("x1", cx).attr("x2", cx)
        .attr("y1", matrixTop + rowH / 2)
        .attr("y2", matrixTop + (rows.length - 1) * rowH + rowH / 2)
        .attr("stroke", present.some(Boolean) ? TEXT_MUTED : GRIDLINE)
        .attr("opacity", 0.4);
      rows.forEach((r, ri) => {
        g.append("circle")
          .attr("cx", cx).attr("cy", matrixTop + ri * rowH + rowH / 2).attr("r", 6)
          .attr("fill", combo[r] ? INCOME_COLOR[r] : "none")
          .attr("stroke", combo[r] ? "none" : GRIDLINE)
          .attr("stroke-width", 1.5);
      });
      g.append("rect")
        .attr("x", x(ci)).attr("y", 0).attr("width", x.bandwidth())
        .attr("height", matrixTop + rows.length * rowH)
        .attr("fill", "transparent")
        .on("mousemove", (evt) => {
          const membership = rows.filter((r) => combo[r]).map((r) => `<div class="tt-row"><span style="color:${INCOME_COLOR[r]}">● ${r}</span></div>`).join("");
          showTooltip(`<div class="tt-title">${fmtCompact(combo.count)} publications</div>${membership}`, evt);
        })
        .on("mouseleave", hideTooltip);
    });
  }

  // ---------------------------------------------------------------------
  // Network stat tiles
  // ---------------------------------------------------------------------
  const VARIANT_LABELS = {
    network_full: "full period, 2015–2026",
    network_early_period: "early period, 2015–2019",
    network_lic_first_author: "low-income first-author only",
  };

  function updateNetworkVariantLabel() {
    document.getElementById("network-stats-variant-label").textContent = VARIANT_LABELS[state.networkVariant];
  }

  // ---------------------------------------------------------------------
  // Network structure over time (cumulative, year by year)
  // ---------------------------------------------------------------------
  function drawTsChart(svgId, seriesSpecs, opts = {}) {
    const rows = opts.rows || DATA.network_yearly_stats;
    const el = document.getElementById(svgId);
    if (!el || !rows || !rows.length) return;
    const width = el.clientWidth || 260;
    const height = +el.getAttribute("height");
    const hasRight = seriesSpecs.some((sp) => sp.axis === "right");
    const margin = { top: 10, right: hasRight ? 32 : 10, bottom: 20, left: 32 };
    const innerW = width - margin.left - margin.right;
    const innerH = height - margin.top - margin.bottom;

    const s = d3.select("#" + svgId).attr("viewBox", `0 0 ${width} ${height}`);
    s.selectAll("*").remove();
    const g = s.append("g").attr("transform", `translate(${margin.left},${margin.top})`);

    const years = rows.map((r) => r.year);
    const x = d3.scaleLinear().domain(d3.extent(years)).range([8, innerW]);

    const leftSpecs = seriesSpecs.filter((sp) => sp.axis !== "right");
    const rightSpecs = seriesSpecs.filter((sp) => sp.axis === "right");
    const scaleFor = (specs, domain) => d3.scaleLinear()
      .domain(domain || [0, d3.max(rows, (r) => d3.max(specs.map((sp) => r[sp.key]))) * 1.1 || 1]).nice()
      .range([innerH, 0]);
    // opts.yDomain fits the axis to the data instead of anchoring at zero.
    // Only for index-style line series (Gini, centralisation) where the whole
    // signal is a movement of a few hundredths -- never for bars.
    const yLeft = scaleFor(leftSpecs, opts.yDomain);
    const yRight = rightSpecs.length ? scaleFor(rightSpecs) : null;

    g.append("g").selectAll("line").data(yLeft.ticks(4)).join("line")
      .attr("class", "gridline").attr("x1", 0).attr("x2", innerW).attr("y1", yLeft).attr("y2", yLeft);
    g.append("g").attr("class", "axis").attr("transform", `translate(0,${innerH})`)
      .call(d3.axisBottom(x).tickValues(years.filter((yr) => yr % 2 === 1 && yr !== years[0])).tickFormat(d3.format("d")).tickSize(0))
      .call((sel) => sel.select(".domain").attr("stroke", BASELINE));
    g.append("g").attr("class", "axis")
      .call(d3.axisLeft(yLeft).ticks(4).tickSize(0).tickFormat(opts.leftFormat || fmtCompact))
      .call((sel) => sel.select(".domain").remove());
    if (yRight) {
      g.append("g").attr("class", "axis").attr("transform", `translate(${innerW},0)`)
        .call(d3.axisRight(yRight).ticks(4).tickSize(0).tickFormat(opts.rightFormat || ((d) => d.toFixed(2))))
        .call((sel) => sel.select(".domain").remove());
    }

    const lineFor = (yScale, key) => d3.line()
      .defined((d) => d[key] !== null && d[key] !== undefined && !Number.isNaN(d[key]))
      .x((d) => x(d.year)).y((d) => yScale(d[key])).curve(d3.curveMonotoneX);

    seriesSpecs.forEach((sp) => {
      const yScale = sp.axis === "right" ? yRight : yLeft;
      g.append("path").attr("fill", "none").attr("stroke", sp.color).attr("stroke-width", 2)
        .attr("d", lineFor(yScale, sp.key)(rows));
      const defined = rows.filter((r) => r[sp.key] !== null && r[sp.key] !== undefined && !Number.isNaN(r[sp.key]));
      if (!defined.length) return;
      const last = defined[defined.length - 1];
      g.append("circle").attr("cx", x(last.year)).attr("cy", yScale(last[sp.key])).attr("r", 3.5)
        .attr("fill", sp.color).attr("stroke", SURFACE).attr("stroke-width", 1.5);
    });

    if (opts.legendId) {
      const legendEl = document.getElementById(opts.legendId);
      if (legendEl && seriesSpecs.length > 1) {
        legendEl.innerHTML = seriesSpecs.map((sp) =>
          `<span class="legend-item"><span class="swatch" style="background:${sp.color}"></span>${sp.label}</span>`
        ).join("");
      } else if (legendEl) {
        legendEl.innerHTML = "";
      }
    }

    // crosshair + tooltip
    const focusLine = g.append("line").attr("class", "gridline").attr("y1", 0).attr("y2", innerH).style("opacity", 0);
    s.append("rect")
      .attr("x", margin.left).attr("y", margin.top).attr("width", innerW).attr("height", innerH)
      .attr("fill", "transparent")
      .on("mousemove", (evt) => {
        const [mx] = d3.pointer(evt, g.node());
        const yr = Math.round(x.invert(mx));
        const row = rows.find((r) => r.year === yr);
        if (!row) return;
        focusLine.attr("x1", x(yr)).attr("x2", x(yr)).style("opacity", 1);
        const html = `<div class="tt-title">${yr}</div>` + seriesSpecs.map((sp) => {
          const v = row[sp.key];
          const fmt = sp.axis === "right" ? (opts.rightFormat || ((d) => d.toFixed(2))) : (opts.leftFormat || fmtCompact);
          return `<div class="tt-row"><span style="color:${sp.color}">● ${sp.label}</span><span>${v === null || v === undefined || Number.isNaN(v) ? "—" : fmt(v)}</span></div>`;
        }).join("");
        showTooltip(html, evt);
      })
      .on("mouseleave", () => { focusLine.style("opacity", 0); hideTooltip(); });
  }

  function renderNetworkTimeSeries() {
    drawTsChart("chart-ts-nodes-density", [
      { key: "Nodes", label: "Nodes", color: SEQ_ACCENT, axis: "left" },
      { key: "Density", label: "Density", color: DIV_NEG, axis: "right" },
    ], { leftFormat: fmtCompact, rightFormat: (d) => d.toFixed(2), legendId: "ts-legend-nodes-density" });

    drawTsChart("chart-ts-smallworld", [
      { key: "Small-World Coefficient", label: "Small-world", color: SEQ_ACCENT, axis: "left" },
    ], { leftFormat: (d) => d.toFixed(1), legendId: "ts-legend-smallworld" });

    drawTsChart("chart-ts-clustering", [
      { key: "Average Clustering", label: "Avg. clustering", color: SEQ_ACCENT, axis: "left" },
      { key: "Modularity Score", label: "Modularity", color: DIV_NEG, axis: "left" },
    ], { leftFormat: (d) => d.toFixed(2), legendId: "ts-legend-clustering" });

    drawTsChart("chart-ts-homophily", [
      { key: "Homophily on Income group", label: "Income", color: SEQ_ACCENT, axis: "left" },
      { key: "Homophily on Region", label: "Region", color: DIV_NEG, axis: "left" },
    ], { leftFormat: (d) => d.toFixed(2), legendId: "ts-legend-homophily" });
  }

  // ---------------------------------------------------------------------
  // Concentration: collaboration topology, within-bloc inequality, and
  // centralisation over time.
  //
  // Every function here no-ops unless BOTH the payload key and the target
  // element exist, so a field whose data.js predates these keys -- or a page
  // that chooses not to show the panel -- renders exactly as before.
  // ---------------------------------------------------------------------
  const BUCKET_COLOR = () => ({ "High only": DIV_POS, "Developing only": DIV_NEG });

  function renderTopologyChart() {
    const el = document.getElementById("chart-topology");
    const topo = DATA.collaboration_topology;
    if (!el || !topo) return;

    const byBucket = new Map(topo.buckets.map((b) => [b.bucket, b]));
    const hi = byBucket.get("High only");
    const dev = byBucket.get("Developing only");
    if (!hi || !dev) return;

    const colors = BUCKET_COLOR();
    const bars = [
      { label: "High-income ↔ high-income", short: "North–North", value: hi.multi_country,
        pct: hi.multi_country_pct, total: hi.works, color: colors["High only"] },
      { label: "LMIC ↔ LMIC", short: "South–South", value: dev.multi_country,
        pct: dev.multi_country_pct, total: dev.works, color: colors["Developing only"] },
    ];

    const width = el.clientWidth || 900;
    const height = +el.getAttribute("height");
    const margin = { top: 10, right: 96, bottom: 26, left: 168 };
    const innerW = width - margin.left - margin.right;
    const innerH = height - margin.top - margin.bottom;

    const s = d3.select("#chart-topology").attr("viewBox", `0 0 ${width} ${height}`);
    s.selectAll("*").remove();
    const g = s.append("g").attr("transform", `translate(${margin.left},${margin.top})`);

    const x = d3.scaleLinear().domain([0, d3.max(bars, (b) => b.value) * 1.05]).nice().range([0, innerW]);
    const y = d3.scaleBand().domain(bars.map((b) => b.short)).range([0, innerH]).padding(0.38);

    g.append("g").selectAll("line").data(x.ticks(5)).join("line")
      .attr("class", "gridline").attr("x1", x).attr("x2", x).attr("y1", 0).attr("y2", innerH);
    g.append("line").attr("class", "baseline").attr("x1", 0).attr("x2", 0).attr("y1", 0).attr("y2", innerH);

    g.append("g").attr("class", "axis").attr("transform", `translate(0,${innerH})`)
      .call(d3.axisBottom(x).ticks(5).tickFormat(fmtCompact).tickSize(0))
      .call((sel) => sel.select(".domain").remove());

    const row = g.selectAll(".topo-row").data(bars).join("g")
      .attr("class", "topo-row")
      .attr("transform", (d) => `translate(0,${y(d.short)})`);

    // 4px rounded data-end, anchored to the baseline at x = 0
    row.append("rect")
      .attr("x", 0).attr("y", 0)
      .attr("width", (d) => Math.max(2, x(d.value)))
      .attr("height", y.bandwidth())
      .attr("rx", 4)
      .attr("fill", (d) => d.color);

    row.append("text").attr("class", "topo-label")
      .attr("x", -12).attr("y", y.bandwidth() / 2)
      .attr("dy", "-0.15em").attr("text-anchor", "end")
      .text((d) => d.short);
    row.append("text").attr("class", "topo-sublabel")
      .attr("x", -12).attr("y", y.bandwidth() / 2)
      .attr("dy", "1.05em").attr("text-anchor", "end")
      .text((d) => d.label);

    row.append("text").attr("class", "topo-value")
      .attr("x", (d) => Math.max(2, x(d.value)) + 10)
      .attr("y", y.bandwidth() / 2).attr("dy", "0.35em")
      .text((d) => d3.format(",")(d.value));

    row.append("rect")
      .attr("x", 0).attr("y", 0).attr("width", innerW).attr("height", y.bandwidth())
      .attr("fill", "transparent")
      .on("mousemove", (evt, d) => {
        showTooltip(
          `<div class="tt-title">${d.short}</div>
           <div class="tt-row"><span style="color:${d.color}">● Multi-country works</span><span>${d3.format(",")(d.value)}</span></div>
           <div class="tt-row"><span>Single-country works</span><span>${d3.format(",")(d.total - d.value)}</span></div>
           <div class="tt-row"><span>Share collaborative</span><span>${d.pct}%</span></div>`,
          evt
        );
      })
      .on("mouseleave", hideTooltip);

    const note = document.getElementById("topology-note");
    if (!note) return;

    const tail = `Only <strong>${dev.multi_country_pct}%</strong> of works with no high-income author span more
      than one country &mdash; the rest is domestic output, not South&ndash;South collaboration.`;

    // Below ~30 South-South works the ratio is dominated by its denominator (a
    // single paper moves it by tens), so state the counts instead of a
    // multiplier that would read as far more precise than it is.
    const MIN_FOR_RATIO = 30;
    if (dev.multi_country >= MIN_FOR_RATIO) {
      const ratio = hi.multi_country / dev.multi_country;
      note.innerHTML = `High-income countries co-author with each other <strong>${ratio.toFixed(0)}×</strong> more
        often than LMICs co-author with each other. ${tail}`;
    } else if (dev.multi_country > 0) {
      note.innerHTML = `Across the whole period, just <strong>${d3.format(",")(dev.multi_country)}</strong>
        publications joined two or more LMICs, against <strong>${d3.format(",")(hi.multi_country)}</strong> joining
        two or more high-income countries. ${tail}`;
    } else {
      note.innerHTML = `Across the whole period, <strong>no</strong> publication joined two or more LMICs, against
        <strong>${d3.format(",")(hi.multi_country)}</strong> joining two or more high-income countries. ${tail}`;
    }
  }

  function renderBlocConcentration() {
    const host = document.getElementById("bloc-concentration");
    const blocs = DATA.bloc_concentration;
    if (!host || !blocs || !blocs.length) return;

    const colors = { "High income": DIV_POS, "Non-high income": DIV_NEG };
    const metrics = [
      { key: "gini", label: "Gini of output", fmt: (v) => v.toFixed(3) },
      { key: "top1_pct", label: "Top country", fmt: (v, b) => `${v}%`, sub: (b) => b.top1_economy },
      { key: "top3_pct", label: "Top 3 countries", fmt: (v) => `${v}%` },
      { key: "countries", label: "Countries contributing", fmt: (v) => d3.format(",")(v) },
    ];

    host.innerHTML = blocs.map((b) => `
      <div class="bloc-card">
        <p class="bloc-title"><span class="swatch" style="background:${colors[b.bloc] || SEQ_ACCENT}"></span>${b.bloc}</p>
        ${metrics.map((m) => `
          <div class="bloc-metric">
            <span class="bloc-metric-label">${m.label}</span>
            <span class="bloc-metric-value">${m.fmt(b[m.key], b)}${m.sub ? `<span class="bloc-metric-sub">${m.sub(b)}</span>` : ""}</span>
          </div>`).join("")}
      </div>`).join("");
  }

  function renderConcentrationTimeSeries() {
    const rows = DATA.concentration_timeseries;
    if (!rows || !rows.length) return;

    // Fit the axis to the data, padded, and clamped to the index's own [0,1]
    // bounds -- these series move by hundredths, which a zero-anchored axis
    // flattens into a straight line.
    const fitDomain = (keys, pad = 0.04) => {
      const vals = rows.flatMap((r) => keys.map((k) => r[k])).filter((v) => v !== null && v !== undefined && !Number.isNaN(v));
      if (!vals.length) return undefined;
      return [Math.max(0, d3.min(vals) - pad), Math.min(1, d3.max(vals) + pad)];
    };

    drawTsChart("chart-ts-gini", [
      { key: "gini_high_income", label: "High income", color: DIV_POS, axis: "left" },
      { key: "gini_non_high_income", label: "Non-high income", color: DIV_NEG, axis: "left" },
    ], {
      rows, leftFormat: (d) => d.toFixed(2), legendId: "ts-legend-gini",
      yDomain: fitDomain(["gini_high_income", "gini_non_high_income"]),
    });

    drawTsChart("chart-ts-centralisation", [
      { key: "freeman_centralisation", label: "Freeman centralisation", color: SEQ_ACCENT, axis: "left" },
    ], {
      rows, leftFormat: (d) => d.toFixed(2), legendId: "ts-legend-centralisation",
      yDomain: fitDomain(["freeman_centralisation"]),
    });
  }

  function renderConcentration() {
    // A page can carry the panel markup before its data.js has been
    // regenerated with these keys. Hide the whole section in that case rather
    // than leaving empty headings above blank charts.
    const panel = document.getElementById("panel-concentration");
    const ready = !!(DATA.collaboration_topology && DATA.bloc_concentration && DATA.concentration_timeseries);
    if (panel) panel.style.display = ready ? "" : "none";
    if (!ready) return;

    renderTopologyChart();
    renderBlocConcentration();
    renderConcentrationTimeSeries();
  }

  // ---------------------------------------------------------------------
  // Centrality distributions (small multiples histograms)
  // ---------------------------------------------------------------------
  const CENTRALITY_METRICS = [
    { key: "degree_centrality", title: "Degree centrality", metric: "degree" },
    { key: "betweenness_centrality", title: "Betweenness centrality", metric: "betweenness" },
    { key: "closeness_centrality", title: "Closeness centrality", metric: "closeness" },
    { key: "participation_coefficient", title: "Participation coefficient", metric: "participation" },
  ];

  function renderCentralityDistributions() {
    const nodes = DATA[state.networkVariant].nodes;
    const container = d3.select("#centrality-distributions");
    container.selectAll("*").remove();

    CENTRALITY_METRICS.forEach((m) => {
      const card = container.append("div").attr("class", "sm-card");
      const width = 240, height = 130, margin = { top: 6, right: 8, bottom: 18, left: 8 };
      const innerW = width - margin.left - margin.right, innerH = height - margin.top - margin.bottom;
      const s = card.append("svg").attr("viewBox", `0 0 ${width} ${height}`).style("width", "100%").style("height", "auto");
      const g = s.append("g").attr("transform", `translate(${margin.left},${margin.top})`);

      const values = nodes.map((n) => n[m.key]).filter((v) => v !== null && v !== undefined && !Number.isNaN(v));
      const x = d3.scaleLinear().domain([0, 1]).range([0, innerW]);
      const bins = d3.bin().domain([0, 1]).thresholds(12)(values);
      const y = d3.scaleLinear().domain([0, d3.max(bins, (b) => b.length) || 1]).range([innerH, 0]);

      g.selectAll("rect").data(bins).join("rect")
        .attr("x", (d) => x(d.x0) + 1)
        .attr("width", (d) => Math.max(0, x(d.x1) - x(d.x0) - 2))
        .attr("y", (d) => y(d.length))
        .attr("height", (d) => innerH - y(d.length))
        .attr("rx", 2)
        .attr("fill", SEQ_ACCENT);

      g.append("g").attr("class", "axis").attr("transform", `translate(0,${innerH})`)
        .call(d3.axisBottom(x).ticks(3).tickSize(0))
        .call((sel) => sel.select(".domain").attr("stroke", BASELINE));

      card.append("p").attr("class", "sm-title")
        .html(`${m.title} <span class="info-icon" data-metric="${m.metric}">&#9432;</span>`);
    });
  }

  // ---------------------------------------------------------------------
  // Bump charts: full year-by-year rank trajectories
  // ---------------------------------------------------------------------
  const BUMP_METRICS = [
    { key: "degree", title: "Degree centrality" },
    { key: "betweenness", title: "Betweenness centrality" },
    { key: "closeness", title: "Closeness centrality" },
  ];
  const BUMP_RANK_CAP = 25;
  const TOP_N_BUMP = 10;

  function drawBumpTrajectoryChart(svgId, metricKey) {
    const bc = DATA.bump_charts_multi;
    const metric = bc[metricKey];
    const el = document.getElementById(svgId);
    const width = el.clientWidth || 520;
    const height = +el.getAttribute("height");
    const margin = { top: 12, right: 108, bottom: 22, left: 92 };
    const innerW = width - margin.left - margin.right;
    const innerH = height - margin.top - margin.bottom;

    const s = d3.select("#" + svgId).attr("viewBox", `0 0 ${width} ${height}`);
    s.selectAll("*").remove();
    const g = s.append("g").attr("transform", `translate(${margin.left},${margin.top})`);

    const years = d3.range(bc.first_year, bc.last_year + 1);
    const x = d3.scaleLinear().domain([bc.first_year, bc.last_year]).range([6, innerW - 6]);
    // Piecewise: the top-10 band gets most of the chart's height, ranks
    // 11-25 are compressed into what's left -- the story is who's in the
    // top 10, so that's where the vertical resolution should go.
    const TOP_BAND_FRACTION = 0.64;
    const y = d3.scaleLinear()
      .domain([1, TOP_N_BUMP, BUMP_RANK_CAP])
      .range([0, innerH * TOP_BAND_FRACTION, innerH]);
    const clampRank = (r) => Math.min(r, BUMP_RANK_CAP);
    const RANK_TICKS = [1, 5, TOP_N_BUMP, 15, 20, BUMP_RANK_CAP];

    g.append("g").selectAll("line").data(RANK_TICKS).join("line")
      .attr("class", "gridline").attr("x1", 0).attr("x2", innerW).attr("y1", y).attr("y2", y);
    g.append("line")
      .attr("x1", 0).attr("x2", innerW).attr("y1", y(TOP_N_BUMP)).attr("y2", y(TOP_N_BUMP))
      .attr("stroke", BASELINE).attr("stroke-width", 1).attr("stroke-dasharray", "2,2");
    g.append("g").attr("class", "axis").attr("transform", `translate(0,${innerH})`)
      .call(d3.axisBottom(x).tickValues(years.filter((yr) => yr % 2 === 1 && yr !== years[0])).tickFormat(d3.format("d")).tickSize(0))
      .call((sel) => sel.select(".domain").attr("stroke", BASELINE));

    // Crosshair + click-to-clear background, appended before the series so
    // it sits *underneath* the lines/hit-paths in z-order -- otherwise this
    // full-area rect would swallow every click and hover meant for a line.
    let selected = null; // declared early; assigned by applyHighlight/click handlers below
    const focusLine = g.append("line").attr("class", "gridline").attr("y1", 0).attr("y2", innerH).style("opacity", 0);
    const bgRect = s.append("rect")
      .attr("x", margin.left).attr("y", margin.top).attr("width", innerW).attr("height", innerH)
      .attr("fill", "transparent")
      .on("mousemove", (evt) => {
        const [mx] = d3.pointer(evt, g.node());
        const yr = Math.round(x.invert(mx));
        if (yr < bc.first_year || yr > bc.last_year) return;
        focusLine.attr("x1", x(yr)).attr("x2", x(yr)).style("opacity", 1);
        const rows = metric.series.filter((d) => d.year === yr).sort((a, b) => a.rank - b.rank);
        const html = `<div class="tt-title">${yr}</div>` + rows.map((d) =>
          `<div class="tt-row"><span style="color:${INCOME_COLOR[d.income_group] || TEXT_MUTED}">● ${d.economy}</span><span>${d.rank}</span></div>`
        ).join("");
        showTooltip(html, evt);
      })
      .on("mouseleave", () => { focusLine.style("opacity", 0); hideTooltip(); });

    const byCountry = d3.group(metric.series, (d) => d.country);
    const lineFor = d3.line().x((d) => x(d.year)).y((d) => y(clampRank(d.rank))).curve(d3.curveMonotoneX);

    const lastSet = new Set(metric.last_top10);
    const firstOnlySet = new Set(metric.first_top10.filter((c) => !lastSet.has(c)));
    const rightLabels = [];
    const leftLabels = [];
    const seriesGroups = []; // { country, economy, color, g }

    byCountry.forEach((rowsRaw, country) => {
      const rows = rowsRaw.slice().sort((a, b) => a.year - b.year);
      const color = INCOME_COLOR[rows[0].income_group] || TEXT_MUTED;

      const seriesG = g.append("g").attr("class", "bump-series").attr("data-country", country);
      seriesG.append("path").attr("class", "bump-line").attr("fill", "none").attr("stroke", color).attr("stroke-width", 2)
        .attr("d", lineFor(rows));
      seriesG.selectAll(null).data(rows).join("circle")
        .attr("cx", (d) => x(d.year)).attr("cy", (d) => y(clampRank(d.rank))).attr("r", 2.2)
        .attr("fill", color);
      // wide invisible hit-path so a thin 2px line is still easy to click/hover
      seriesG.append("path").attr("class", "bump-hit").attr("fill", "none").attr("stroke", "#000")
        .attr("stroke-width", 10).attr("opacity", 0).style("cursor", "pointer")
        .attr("d", lineFor(rows));

      seriesGroups.push({ country, economy: rows[0].economy, color, el: seriesG });

      if (lastSet.has(country)) {
        const last = rows[rows.length - 1];
        rightLabels.push({ country, economy: last.economy, rank: last.rank, y: y(clampRank(last.rank)), color });
      }
      if (firstOnlySet.has(country)) {
        const first = rows[0];
        leftLabels.push({ country, economy: first.economy, rank: first.rank, y: y(clampRank(first.rank)), color });
      }
    });

    function placeLabels(labels, side) {
      labels.sort((a, b) => a.y - b.y);
      for (let i = 1; i < labels.length; i++) {
        if (labels[i].y - labels[i - 1].y < 12) labels[i].y = labels[i - 1].y + 12;
      }
      return g.selectAll(null).data(labels).join("text")
        .attr("class", "direct-label bump-label").attr("data-country", (d) => d.country)
        .style("font-size", "10px").style("cursor", "pointer")
        .attr("text-anchor", side === "right" ? "start" : "end")
        .attr("x", side === "right" ? innerW + 8 : -8)
        .attr("y", (d) => d.y + 3.5)
        .attr("fill", (d) => d.color)
        .text((d) => `${d.rank}. ${d.economy}`);
    }
    // Country names vary a lot in length ("China" vs. "Hong Kong SAR, China");
    // the margins only budget so much room for them. Rather than clip against
    // the SVG's edge (which cuts off the *start* of "end"-anchored left labels),
    // shorten anything that doesn't fit into an ellipsis and expose the full
    // name via <title> so it's still available on hover.
    function truncateToFit(sel, maxWidth) {
      sel.each(function (d) {
        const node = this;
        const full = `${d.rank}. ${d.economy}`;
        if (node.getComputedTextLength() <= maxWidth) return;
        let text = full;
        while (text.length > 1 && node.getComputedTextLength() > maxWidth) {
          text = text.slice(0, -1);
          node.textContent = text.trimEnd() + "…";
        }
        d3.select(node).append("title").text(full);
      });
    }
    const rightLabelSel = placeLabels(rightLabels, "right");
    const leftLabelSel = placeLabels(leftLabels, "left");
    truncateToFit(rightLabelSel, margin.right - 14);
    truncateToFit(leftLabelSel, margin.left - 14);

    // Selection badge: names the currently-picked country so a chosen line
    // is unambiguous even where several lines cross or share a color.
    const badge = s.append("text").attr("class", "bump-badge")
      .attr("x", margin.left + innerW).attr("y", margin.top - 2).style("opacity", 0);

    function applyHighlight(country) {
      g.selectAll(".bump-series").style("opacity", (function () {
        return function () {
          const c = d3.select(this).attr("data-country");
          return !country || c === country ? 1 : 0.1;
        };
      })());
      g.selectAll(".bump-line").attr("stroke-width", function () {
        const c = d3.select(this.parentNode).attr("data-country");
        return country && c === country ? 3 : 2;
      });
      rightLabelSel.style("opacity", (d) => (!country || d.country === country ? 1 : 0.15));
      leftLabelSel.style("opacity", (d) => (!country || d.country === country ? 1 : 0.15));
      if (country) {
        const info = seriesGroups.find((sg) => sg.country === country);
        badge.style("opacity", 1).attr("fill", info.color).text(info.economy);
      } else {
        badge.style("opacity", 0);
      }
    }

    seriesGroups.forEach(({ country, el }) => {
      el.on("mouseenter", () => { if (!selected) applyHighlight(country); })
        .on("mouseleave", () => { if (!selected) applyHighlight(null); })
        .on("click", (evt) => {
          evt.stopPropagation();
          selected = selected === country ? null : country;
          applyHighlight(selected);
        });
    });
    rightLabelSel.on("click", (evt, d) => {
      evt.stopPropagation();
      selected = selected === d.country ? null : d.country;
      applyHighlight(selected);
    });
    leftLabelSel.on("click", (evt, d) => {
      evt.stopPropagation();
      selected = selected === d.country ? null : d.country;
      applyHighlight(selected);
    });

    bgRect.on("click", () => { selected = null; applyHighlight(null); });
  }

  function renderBumpCharts() {
    const bc = DATA.bump_charts_multi;
    document.querySelectorAll("#bump-first-label, #bump-first-label-2").forEach((el) => { el.textContent = bc.first_year; });
    document.getElementById("bump-last-label").textContent = bc.last_year;

    d3.select("#bump-legend").html(
      INCOME_ORDER.map((g) => `<span class="legend-item"><span class="swatch" style="background:${INCOME_COLOR[g]}"></span>${g}</span>`).join("")
    );

    const container = d3.select("#bump-charts");
    container.selectAll("*").remove();

    BUMP_METRICS.forEach((m) => {
      const card = container.append("div").attr("class", "bump-card");
      card.append("p").attr("class", "sm-title")
        .html(`${m.title} <span class="info-icon" data-metric="${m.key}">&#9432;</span>`);
      card.append("svg").attr("id", `chart-bump-${m.key}`).attr("class", "chart").attr("width", "100%").attr("height", 280);
      drawBumpTrajectoryChart(`chart-bump-${m.key}`, m.key);
    });
  }

  // ---------------------------------------------------------------------
  // Filter bar wiring
  // ---------------------------------------------------------------------
  function renderIncomeChips() {
    const chips = d3.select("#income-chip-set").selectAll(".chip").data(INCOME_ORDER).join("button")
      .attr("class", "chip")
      .classed("is-active", (d) => state.activeIncome.has(d))
      .classed("is-off", (d) => !state.activeIncome.has(d))
      .html((d) => `<span class="swatch" style="background:${INCOME_COLOR[d]}"></span>${d.replace(" income", "")}`)
      .on("click", (evt, d) => {
        if (state.activeIncome.has(d)) {
          if (state.activeIncome.size > 1) state.activeIncome.delete(d);
        } else {
          state.activeIncome.add(d);
        }
        renderIncomeChips();
        renderMapLayer();
        renderTrendCharts();
      });
  }

  function wireFilterBar() {
    d3.selectAll("#map-mode-toggle button").on("click", function () {
      const mode = this.getAttribute("data-mode");
      state.mapMode = mode;
      d3.selectAll("#map-mode-toggle button").classed("is-active", false);
      d3.select(this).classed("is-active", true);
      document.getElementById("network-variant-group").style.display = mode === "network" ? "flex" : "none";
      document.getElementById("year-range-group").style.display = mode === "choropleth" ? "flex" : "none";
      renderMapLayer();
    });

    d3.select("#network-variant-select").on("change", function () {
      state.networkVariant = this.value;
      renderMapLayer();
      updateNetworkVariantLabel();
      renderCentralityDistributions();
    });

    const minInput = document.getElementById("year-range-min");
    const maxInput = document.getElementById("year-range-max");
    const trackFill = document.getElementById("range-track-fill");
    const domainMin = DATA.summary.year_min;
    const domainMax = DATA.summary.year_max;

    function syncYearRange() {
      const lo = +minInput.value, hi = +maxInput.value;
      // Keep the two thumbs from crossing by constraining each input's
      // opposite bound natively, instead of swapping values on cross (which
      // makes a thumb jump to the other side of the track).
      minInput.max = hi;
      maxInput.min = lo;

      state.yearMin = lo; state.yearMax = hi;
      document.getElementById("year-range-min-label").textContent = lo;
      document.getElementById("year-range-max-label").textContent = hi;

      const pctLo = ((lo - domainMin) / (domainMax - domainMin)) * 100;
      const pctHi = ((hi - domainMin) / (domainMax - domainMin)) * 100;
      trackFill.style.left = pctLo + "%";
      trackFill.style.width = Math.max(0, pctHi - pctLo) + "%";

      if (state.mapMode === "choropleth") renderChoropleth();
    }
    minInput.min = maxInput.min = domainMin;
    minInput.max = maxInput.max = domainMax;
    minInput.value = domainMin;
    maxInput.value = domainMax;
    syncYearRange();

    minInput.addEventListener("input", syncYearRange);
    maxInput.addEventListener("input", syncYearRange);
  }

  function wireZoomControls() {
    svg.call(zoomBehavior);
    document.getElementById("zoom-in").addEventListener("click", () => {
      svg.transition().duration(200).call(zoomBehavior.scaleBy, 1.6);
    });
    document.getElementById("zoom-out").addEventListener("click", () => {
      svg.transition().duration(200).call(zoomBehavior.scaleBy, 1 / 1.6);
    });
    document.getElementById("zoom-reset").addEventListener("click", () => {
      svg.transition().duration(250).call(zoomBehavior.transform, d3.zoomIdentity);
    });
  }

  // ---------------------------------------------------------------------
  // Theme toggle
  // ---------------------------------------------------------------------
  function currentTheme() {
    const stamped = document.documentElement.getAttribute("data-theme");
    if (stamped === "light" || stamped === "dark") return stamped;
    return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  }

  function updateThemeButton(theme) {
    const btn = document.getElementById("theme-toggle");
    // Icon shows the theme a click switches TO, not the current one.
    btn.textContent = theme === "dark" ? "☀" : "☾";
    const label = theme === "dark" ? "Switch to light theme" : "Switch to dark theme";
    btn.title = label;
    btn.setAttribute("aria-label", label);
  }

  function applyTheme(theme, { rerender } = { rerender: true }) {
    document.documentElement.setAttribute("data-theme", theme);
    try { localStorage.setItem("theme", theme); } catch (e) {}
    updateThemeButton(theme);
    if (!rerender) return;

    // CSS repaints panels/text instantly via var(); the SVG charts below
    // cached their colors from those same variables at render time, so they
    // need a fresh token read + redraw to pick up the new theme. Zoom/pan
    // and filter state are left untouched -- only colors are refreshed.
    refreshColorTokens();
    renderIncomeChips();
    renderMapLayer();
    renderTrendCharts();
    renderEquityChart();
    renderUpsetChart();
    renderConcentration();
    renderNetworkTimeSeries();
    renderCentralityDistributions();
    renderBumpCharts();
  }

  function wireThemeToggle() {
    updateThemeButton(currentTheme());
    document.getElementById("theme-toggle").addEventListener("click", () => {
      applyTheme(currentTheme() === "dark" ? "light" : "dark");
    });
  }

  // ---------------------------------------------------------------------
  // Init
  // ---------------------------------------------------------------------
  function init() {
    const generatedDate = new Date(DATA.generated_at);
    document.getElementById("generated-at").textContent =
      generatedDate.toLocaleDateString(undefined, { year: "numeric", month: "long", day: "numeric" });
    document.getElementById("footer-cite-year").textContent = `(${generatedDate.getFullYear()})`;

    renderKPIRow();
    renderIncomeChips();
    wireFilterBar();
    wireZoomControls();
    wireThemeToggle();
    drawBaseLand();
    fitProjection();
    renderTrendCharts();
    renderEquityChart();
    renderUpsetChart();
    renderConcentration();
    updateNetworkVariantLabel();
    renderNetworkTimeSeries();
    renderCentralityDistributions();
    renderBumpCharts();

    let resizeTimer;
    window.addEventListener("resize", () => {
      clearTimeout(resizeTimer);
      resizeTimer = setTimeout(() => {
        fitProjection();
        renderTrendCharts();
        renderEquityChart();
        renderUpsetChart();
        renderConcentration();
        renderNetworkTimeSeries();
        renderCentralityDistributions();
        renderBumpCharts();
      }, 150);
    });
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
