const DATA_PATHS = [
  "../data/processed/panel_ine_cno_ai_exposure.csv",
  "./data/processed/panel_ine_cno_ai_exposure.csv",
];

const state = {
  rows: [],
  periods: [],
  years: [],
  selectedPeriod: "",
  selectedExposure: "all",
  selectedStartYear: 2021,
  search: "",
};

const fmtNumber = new Intl.NumberFormat("es-ES", { maximumFractionDigits: 1 });
const fmtPct = new Intl.NumberFormat("es-ES", {
  maximumFractionDigits: 2,
  minimumFractionDigits: 0,
});

function normalizeText(value) {
  return String(value || "")
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .toLowerCase();
}

function parseCsv(text) {
  const rows = [];
  let current = "";
  let row = [];
  let inQuotes = false;

  for (let i = 0; i < text.length; i += 1) {
    const char = text[i];
    const next = text[i + 1];

    if (char === '"' && inQuotes && next === '"') {
      current += '"';
      i += 1;
    } else if (char === '"') {
      inQuotes = !inQuotes;
    } else if (char === "," && !inQuotes) {
      row.push(current);
      current = "";
    } else if ((char === "\n" || char === "\r") && !inQuotes) {
      if (char === "\r" && next === "\n") i += 1;
      row.push(current);
      if (row.some((value) => value !== "")) rows.push(row);
      row = [];
      current = "";
    } else {
      current += char;
    }
  }

  if (current || row.length) {
    row.push(current);
    rows.push(row);
  }

  const headers = rows.shift();
  return rows.map((values) =>
    headers.reduce((record, header, index) => {
      record[header] = values[index] ?? "";
      return record;
    }, {})
  );
}

function toNumber(value) {
  if (value === "" || value === "nan" || value == null) return null;
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function cleanRows(rows) {
  return rows.map((row) => ({
    ...row,
    year: Number(row.year),
    quarter: Number(row.quarter),
    employment_thousand_people: toNumber(row.employment_thousand_people),
    employment_qoq_pct: toNumber(row.employment_qoq_pct),
    employment_yoy_pct: toNumber(row.employment_yoy_pct),
    aioe_score_cno_proxy: toNumber(row.aioe_score_cno_proxy),
    language_modeling_aioe_score_cno_proxy: toNumber(
      row.language_modeling_aioe_score_cno_proxy
    ),
    anthropic_observed_exposure_cno_proxy: toNumber(
      row.anthropic_observed_exposure_cno_proxy
    ),
    aioe_percentile_cno_proxy: toNumber(row.aioe_percentile_cno_proxy),
    language_modeling_percentile_cno_proxy: toNumber(
      row.language_modeling_percentile_cno_proxy
    ),
    anthropic_percentile_cno_proxy: toNumber(row.anthropic_percentile_cno_proxy),
    employment_next_quarter_thousand_people: toNumber(
      row.employment_next_quarter_thousand_people
    ),
    employment_growth_next_quarter_pct: toNumber(
      row.employment_growth_next_quarter_pct
    ),
    ai_exposure_category:
      row.ai_exposure_category && row.ai_exposure_category !== "nan"
        ? row.ai_exposure_category
        : "sin_categoria",
  }));
}

function uniqueSorted(values, desc = false) {
  const unique = [...new Set(values)];
  unique.sort();
  if (desc) unique.reverse();
  return unique;
}

function avg(values) {
  if (!values.length) return null;
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function setupControls() {
  const periodSelect = document.getElementById("periodSelect");
  const startYearSelect = document.getElementById("startYearSelect");

  periodSelect.innerHTML = state.periods
    .map((period) => `<option value="${period}">${period}</option>`)
    .join("");
  periodSelect.value = state.selectedPeriod;

  startYearSelect.innerHTML = state.years
    .map((year) => `<option value="${year}">${year}</option>`)
    .join("");
  startYearSelect.value = String(state.selectedStartYear);

  periodSelect.addEventListener("change", (event) => {
    state.selectedPeriod = event.target.value;
    render();
  });
  document.getElementById("exposureSelect").addEventListener("change", (event) => {
    state.selectedExposure = event.target.value;
    render();
  });
  startYearSelect.addEventListener("change", (event) => {
    state.selectedStartYear = Number(event.target.value);
    render();
  });
  document.getElementById("searchInput").addEventListener("input", (event) => {
    state.search = event.target.value.trim().toLowerCase();
    render();
  });
}

function rowMatchesFilters(row) {
  const matchesExposure =
    state.selectedExposure === "all" || row.ai_exposure_category === state.selectedExposure;
  const matchesSearch = state.search
    ? normalizeText(row.cno_title).includes(normalizeText(state.search))
    : true;
  return matchesExposure && matchesSearch;
}

function getFilteredPeriodRows() {
  return state.rows
    .filter((row) => row.periodo === state.selectedPeriod)
    .filter(rowMatchesFilters);
}

function getTrendRows() {
  return state.rows
    .filter((row) => row.year >= state.selectedStartYear)
    .filter(rowMatchesFilters);
}

function renderKpis(rows) {
  const totalEmployment = rows.reduce(
    (sum, row) => sum + (row.employment_thousand_people || 0),
    0
  );
  const growthValues = rows
    .map((row) => row.employment_yoy_pct)
    .filter((value) => value != null);
  const exposureValues = rows
    .map((row) => row.language_modeling_percentile_cno_proxy)
    .filter((value) => value != null);
  const quarterlyGrowthValues = rows
    .map((row) => row.employment_qoq_pct)
    .filter((value) => value != null);
  const anthropicValues = rows
    .map((row) => row.anthropic_observed_exposure_cno_proxy)
    .filter((value) => value != null);

  document.getElementById("kpiEmployment").textContent = fmtNumber.format(totalEmployment);
  document.getElementById("kpiGrowth").textContent =
    growthValues.length > 0 ? `${fmtPct.format(avg(growthValues))}%` : "-";
  document.getElementById("kpiOccupations").textContent = String(rows.length);
  document.getElementById("kpiExposure").textContent =
    exposureValues.length > 0 ? fmtNumber.format(avg(exposureValues)) : "-";
  document.getElementById("kpiQuarterlyGrowth").textContent =
    quarterlyGrowthValues.length > 0 ? `${fmtPct.format(avg(quarterlyGrowthValues))}%` : "-";
  document.getElementById("kpiAnthropic").textContent =
    anthropicValues.length > 0 ? fmtNumber.format(avg(anthropicValues)) : "-";
}

function colorForCategory(category) {
  if (category === "alta") return "#006b5f";
  if (category === "media") return "#3167a8";
  if (category === "baja") return "#b85c28";
  return "#8a97a3";
}

function clear(node) {
  node.replaceChildren();
}

function svgEl(tag, attrs = {}) {
  const el = document.createElementNS("http://www.w3.org/2000/svg", tag);
  Object.entries(attrs).forEach(([key, value]) => el.setAttribute(key, value));
  return el;
}

function showTooltip(event, html) {
  let tip = document.querySelector(".tooltip");
  if (!tip) {
    tip = document.createElement("div");
    tip.className = "tooltip";
    document.body.appendChild(tip);
  }
  tip.innerHTML = html;
  tip.style.left = `${event.clientX + 14}px`;
  tip.style.top = `${event.clientY + 14}px`;
  tip.hidden = false;
}

function hideTooltip() {
  const tip = document.querySelector(".tooltip");
  if (tip) tip.hidden = true;
}

function renderEmptyState(container, message) {
  clear(container);
  const empty = document.createElement("p");
  empty.className = "empty-state";
  empty.textContent = message;
  container.appendChild(empty);
}

function renderFilterSummary(rows, trendRows) {
  const container = document.getElementById("filterSummary");
  clear(container);

  if (!rows.length) {
    renderEmptyState(container, "No hay datos para el filtro actual.");
    return;
  }

  const trendYears = uniqueSorted(trendRows.map((row) => row.year).filter(Boolean));
  const uniqueValues = (field) => uniqueSorted(rows.map((row) => row[field]).filter(Boolean));
  const sexValues = uniqueValues("sexo");
  const statusValues = uniqueValues("situacion_profesional");
  const socGroups = new Set(
    rows.flatMap((row) => String(row.mapped_soc_majors || "").split(",").map((v) => v.trim()).filter(Boolean))
  );
  const items = [
    ["Sexo", sexValues.length === 1 ? sexValues[0] : `${sexValues.length} valores`],
    ["Situacion profesional", statusValues.length === 1 ? statusValues[0] : `${statusValues.length} valores`],
    ["Grupos CNO seleccionados", String(new Set(rows.map((row) => row.cno_major)).size)],
    ["Periodo de corte", state.selectedPeriod],
    [
      "Rango temporal",
      trendYears.length ? `${trendYears[0]}-${trendYears[trendYears.length - 1]}` : "-",
    ],
    ["Grupos SOC representados", String(socGroups.size)],
    [
      "AIOE medio",
      rows.some((row) => row.aioe_score_cno_proxy != null)
        ? fmtNumber.format(avg(rows.map((row) => row.aioe_score_cno_proxy).filter((v) => v != null)))
        : "-",
    ],
    [
      "LM AIOE medio",
      rows.some((row) => row.language_modeling_aioe_score_cno_proxy != null)
        ? fmtNumber.format(
            avg(
              rows
                .map((row) => row.language_modeling_aioe_score_cno_proxy)
                .filter((v) => v != null)
            )
          )
        : "-",
    ],
    ["Busqueda", state.search || "sin filtro"],
  ];

  items.forEach(([label, value]) => {
    const card = document.createElement("article");
    card.className = "summary-card";
    card.innerHTML = `<span>${label}</span><strong>${value}</strong>`;
    container.appendChild(card);
  });
}

function renderCategoryBreakdown(rows) {
  const container = document.getElementById("categoryBreakdown");
  clear(container);

  if (!rows.length) {
    renderEmptyState(container, "No hay datos para el filtro actual.");
    return;
  }

  const totals = new Map();
  rows.forEach((row) => {
    const current = totals.get(row.ai_exposure_category) || 0;
    totals.set(row.ai_exposure_category, current + (row.employment_thousand_people || 0));
  });

  const data = [...totals.entries()]
    .map(([category, employment]) => ({ category, employment }))
    .sort((a, b) => b.employment - a.employment);
  const grandTotal = data.reduce((sum, row) => sum + row.employment, 0);
  const width = 640;
  const height = Math.max(220, data.length * 62 + 18);
  const margin = { top: 12, right: 24, bottom: 12, left: 22 };
  const barWidth = width - margin.left - margin.right;
  const svg = svgEl("svg", { viewBox: `0 0 ${width} ${height}` });

  data.forEach((row, index) => {
    const y = margin.top + index * 62;
    const share = grandTotal === 0 ? 0 : (row.employment / grandTotal) * 100;

    svg.appendChild(svgEl("text", {
      x: margin.left,
      y: y + 14,
      class: "bar-label",
    })).textContent = `${row.category} · ${fmtPct.format(share)}%`;

    svg.appendChild(svgEl("rect", {
      x: margin.left,
      y: y + 24,
      width: barWidth,
      height: 18,
      rx: 9,
      fill: "#e7edf1",
    }));

    const fill = svgEl("rect", {
      x: margin.left,
      y: y + 24,
      width: Math.max(6, (share / 100) * barWidth),
      height: 18,
      rx: 9,
      fill: colorForCategory(row.category),
    });
    fill.addEventListener("mousemove", (event) =>
      showTooltip(
        event,
        `<strong>${row.category}</strong><br>Empleo: ${fmtNumber.format(
          row.employment
        )} miles<br>Peso: ${fmtPct.format(share)}%`
      )
    );
    fill.addEventListener("mouseleave", hideTooltip);
    svg.appendChild(fill);

    svg.appendChild(svgEl("text", {
      x: margin.left,
      y: y + 54,
      class: "tick-label",
    })).textContent = `${fmtNumber.format(row.employment)} miles`;
  });

  container.appendChild(svg);
}

function renderEmploymentBar(rows) {
  const container = document.getElementById("employmentBar");
  clear(container);
  const data = [...rows].sort(
    (a, b) => (b.employment_thousand_people || 0) - (a.employment_thousand_people || 0)
  );
  document.getElementById("barSubtitle").textContent = state.selectedPeriod;

  if (!data.length) {
    renderEmptyState(container, "No hay datos para el filtro actual.");
    return;
  }

  const width = 1120;
  const height = Math.max(340, data.length * 34 + 70);
  const margin = { top: 12, right: 130, bottom: 30, left: 150 };
  const chartWidth = width - margin.left - margin.right;
  const max = Math.max(...data.map((row) => row.employment_thousand_people || 0), 1);
  const svg = svgEl("svg", { viewBox: `0 0 ${width} ${height}` });

  data.forEach((row, index) => {
    const y = margin.top + index * 34;
    const barWidth = ((row.employment_thousand_people || 0) / max) * chartWidth;

    svg.appendChild(svgEl("text", {
      x: 12,
      y: y + 20,
      class: "bar-label",
    })).textContent = `${row.cno_major}. ${shortOccupationLabel(row.cno_title)}`;

    const rect = svgEl("rect", {
      x: margin.left,
      y,
      width: Math.max(2, barWidth),
      height: 22,
      rx: 4,
      fill: colorForCategory(row.ai_exposure_category),
    });
    rect.addEventListener("mousemove", (event) =>
      showTooltip(
        event,
        `<strong>${row.cno_title}</strong><br>Empleo: ${fmtNumber.format(
          row.employment_thousand_people || 0
        )} miles<br>Var. interanual: ${
          row.employment_yoy_pct == null ? "-" : `${fmtPct.format(row.employment_yoy_pct)}%`
        }<br>IA: ${row.ai_exposure_category}`
      )
    );
    rect.addEventListener("mouseleave", hideTooltip);
    svg.appendChild(rect);

    svg.appendChild(svgEl("text", {
      x: margin.left + barWidth + 8,
      y: y + 16,
      class: "tick-label",
    })).textContent = fmtNumber.format(row.employment_thousand_people || 0);
  });

  container.appendChild(svg);
}

function renderExposureRanking(rows) {
  const container = document.getElementById("exposureRanking");
  clear(container);

  const latestByCno = new Map();
  rows.forEach((row) => latestByCno.set(row.cno_major, row));
  const data = [...latestByCno.values()]
    .filter((row) => row.language_modeling_percentile_cno_proxy != null)
    .sort(
      (a, b) =>
        b.language_modeling_percentile_cno_proxy - a.language_modeling_percentile_cno_proxy
    );

  if (!data.length) {
    renderEmptyState(container, "No hay percentiles de IA disponibles.");
    return;
  }

  const width = 640;
  const height = 330;
  const margin = { top: 14, right: 28, bottom: 30, left: 150 };
  const chartWidth = width - margin.left - margin.right;
  const svg = svgEl("svg", { viewBox: `0 0 ${width} ${height}` });

  data.forEach((row, index) => {
    const y = margin.top + index * 30;
    const value = row.language_modeling_percentile_cno_proxy || 0;
    const barWidth = (value / 100) * chartWidth;

    svg.appendChild(svgEl("text", {
      x: 8,
      y: y + 17,
      class: "tick-label",
    })).textContent = `${row.cno_major}. ${shortOccupationLabel(row.cno_title)}`;

    svg.appendChild(svgEl("rect", {
      x: margin.left,
      y,
      width: Math.max(2, barWidth),
      height: 20,
      rx: 4,
      fill: colorForCategory(row.ai_exposure_category),
    }));

    svg.appendChild(svgEl("text", {
      x: margin.left + barWidth + 8,
      y: y + 15,
      class: "tick-label",
    })).textContent = fmtNumber.format(value);
  });

  container.appendChild(svg);
}

function renderScatter(rows) {
  const container = document.getElementById("scatterPlot");
  clear(container);
  const data = rows.filter(
    (row) =>
      row.language_modeling_percentile_cno_proxy != null && row.employment_yoy_pct != null
  );

  if (!data.length) {
    renderEmptyState(container, "No hay observaciones suficientes para la dispersion.");
    return;
  }

  const width = 640;
  const height = 330;
  const margin = { top: 20, right: 26, bottom: 48, left: 58 };
  const xMin = 0;
  const xMax = 100;
  const yValues = data.map((row) => row.employment_yoy_pct);
  const yMin = Math.min(-8, ...yValues);
  const yMax = Math.max(8, ...yValues);
  const plotWidth = width - margin.left - margin.right;
  const plotHeight = height - margin.top - margin.bottom;
  const x = (value) => margin.left + ((value - xMin) / (xMax - xMin)) * plotWidth;
  const y = (value) => margin.top + (1 - (value - yMin) / (yMax - yMin)) * plotHeight;
  const svg = svgEl("svg", { viewBox: `0 0 ${width} ${height}` });

  [0, 25, 50, 75, 100].forEach((tick) => {
    svg.appendChild(svgEl("line", {
      x1: x(tick),
      x2: x(tick),
      y1: margin.top,
      y2: height - margin.bottom,
      class: "grid-line",
    }));
    svg.appendChild(svgEl("text", {
      x: x(tick),
      y: height - 18,
      class: "tick-label",
      "text-anchor": "middle",
    })).textContent = tick;
  });

  [yMin, 0, yMax].forEach((tick) => {
    svg.appendChild(svgEl("line", {
      x1: margin.left,
      x2: width - margin.right,
      y1: y(tick),
      y2: y(tick),
      class: "grid-line",
    }));
    svg.appendChild(svgEl("text", {
      x: 8,
      y: y(tick) + 4,
      class: "tick-label",
    })).textContent = `${fmtPct.format(tick)}%`;
  });

  data.forEach((row) => {
    const circle = svgEl("circle", {
      cx: x(row.language_modeling_percentile_cno_proxy),
      cy: y(row.employment_yoy_pct),
      r: 8,
      fill: colorForCategory(row.ai_exposure_category),
      opacity: 0.86,
    });
    circle.addEventListener("mousemove", (event) =>
      showTooltip(
        event,
        `<strong>${row.cno_title}</strong><br>Percentil IA: ${fmtNumber.format(
          row.language_modeling_percentile_cno_proxy
        )}<br>Var. interanual: ${fmtPct.format(row.employment_yoy_pct)}%`
      )
    );
    circle.addEventListener("mouseleave", hideTooltip);
    svg.appendChild(circle);
  });

  svg.appendChild(svgEl("text", {
    x: width / 2,
    y: height - 2,
    class: "axis-label",
    "text-anchor": "middle",
  })).textContent = "Percentil de exposicion a modelos de lenguaje";
  svg.appendChild(svgEl("text", {
    x: 12,
    y: 14,
    class: "axis-label",
  })).textContent = "Var. interanual";

  container.appendChild(svg);
}

function renderTrend(rows) {
  const container = document.getElementById("trendChart");
  clear(container);

  const grouped = new Map();
  rows
    .filter((row) => row.ai_exposure_category !== "sin_categoria")
    .forEach((row) => {
      const key = `${row.periodo}|${row.ai_exposure_category}`;
      const current = grouped.get(key) || {
        periodo: row.periodo,
        year: row.year,
        quarter: row.quarter,
        category: row.ai_exposure_category,
        employment: 0,
      };
      current.employment += row.employment_thousand_people || 0;
      grouped.set(key, current);
    });

  const categories = ["alta", "media", "baja"].filter((category) =>
    [...grouped.values()].some((row) => row.category === category)
  );
  const periods = uniqueSorted([...new Set([...grouped.values()].map((row) => row.periodo))]);
  const series = categories.map((category) => {
    const values = periods
      .map((period) => grouped.get(`${period}|${category}`))
      .filter(Boolean);
    const base = values[0]?.employment || 1;
    return {
      category,
      values: values.map((row) => ({ ...row, index: (row.employment / base) * 100 })),
    };
  });

  if (!series.length) {
    renderEmptyState(
      container,
      "No hay series temporales disponibles para el filtro actual."
    );
    return;
  }

  const allValues = series.flatMap((item) => item.values.map((row) => row.index));
  const width = 1120;
  const height = 380;
  const margin = { top: 24, right: 120, bottom: 46, left: 58 };
  const plotWidth = width - margin.left - margin.right;
  const plotHeight = height - margin.top - margin.bottom;
  const yMin = Math.min(90, ...allValues);
  const yMax = Math.max(112, ...allValues);
  const x = (index) => margin.left + (index / Math.max(1, periods.length - 1)) * plotWidth;
  const y = (value) => margin.top + (1 - (value - yMin) / (yMax - yMin)) * plotHeight;
  const svg = svgEl("svg", { viewBox: `0 0 ${width} ${height}` });

  [yMin, 100, yMax].forEach((tick) => {
    svg.appendChild(svgEl("line", {
      x1: margin.left,
      x2: width - margin.right,
      y1: y(tick),
      y2: y(tick),
      class: "grid-line",
    }));
    svg.appendChild(svgEl("text", {
      x: 10,
      y: y(tick) + 4,
      class: "tick-label",
    })).textContent = fmtNumber.format(tick);
  });

  series.forEach((serie) => {
    const path = serie.values
      .map(
        (row, index) =>
          `${index === 0 ? "M" : "L"} ${x(periods.indexOf(row.periodo))} ${y(row.index)}`
      )
      .join(" ");
    svg.appendChild(svgEl("path", {
      d: path,
      fill: "none",
      stroke: colorForCategory(serie.category),
      "stroke-width": 3,
      "stroke-linecap": "round",
      "stroke-linejoin": "round",
    }));
    const last = serie.values[serie.values.length - 1];
    if (last) {
      svg.appendChild(svgEl("text", {
        x: width - margin.right + 10,
        y: y(last.index) + 4,
        class: "legend-text",
      })).textContent = `${serie.category} ${fmtNumber.format(last.index)}`;
    }
  });

  periods
    .filter((_, index) => index % 4 === 0 || index === periods.length - 1)
    .forEach((period) => {
      const index = periods.indexOf(period);
      svg.appendChild(svgEl("text", {
        x: x(index),
        y: height - 16,
        class: "tick-label",
        "text-anchor": "middle",
      })).textContent = period;
    });

  container.appendChild(svg);
}

function renderTable(rows) {
  const tbody = document.getElementById("dataTable");
  clear(tbody);
  document.getElementById("tableCount").textContent = `${rows.length} filas`;

  rows
    .sort((a, b) => (b.employment_thousand_people || 0) - (a.employment_thousand_people || 0))
    .forEach((row) => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td>${row.periodo}</td>
        <td>${row.cno_major}</td>
        <td>${row.cno_title}</td>
        <td class="numeric">${fmtNumber.format(row.employment_thousand_people || 0)}</td>
        <td class="numeric">${
          row.employment_qoq_pct == null ? "-" : `${fmtPct.format(row.employment_qoq_pct)}%`
        }</td>
        <td class="numeric">${
          row.employment_yoy_pct == null ? "-" : `${fmtPct.format(row.employment_yoy_pct)}%`
        }</td>
        <td>${row.ai_exposure_category}</td>
        <td class="numeric">${row.mapped_soc_majors || "-"}</td>
        <td class="numeric">${
          row.aioe_score_cno_proxy == null ? "-" : fmtNumber.format(row.aioe_score_cno_proxy)
        }</td>
        <td class="numeric">${
          row.aioe_percentile_cno_proxy == null
            ? "-"
            : fmtNumber.format(row.aioe_percentile_cno_proxy)
        }</td>
        <td class="numeric">${
          row.language_modeling_percentile_cno_proxy == null
            ? "-"
            : fmtNumber.format(row.language_modeling_percentile_cno_proxy)
        }</td>
        <td class="numeric">${
          row.anthropic_percentile_cno_proxy == null
            ? "-"
            : fmtNumber.format(row.anthropic_percentile_cno_proxy)
        }</td>
      `;
      tbody.appendChild(tr);
    });
}

function shortLabel(value, maxLength) {
  return value.length <= maxLength ? value : `${value.slice(0, maxLength - 1)}...`;
}

function shortOccupationLabel(value) {
  const labels = [
    ["Directores y gerentes", "Directivos"],
    ["Tecnicos y profesionales cientificos e intelectuales", "Profesionales"],
    ["Tecnicos, profesionales de apoyo", "Tecnicos apoyo"],
    ["Empleados contables, administrativos y otros empleados de oficina", "Administrativos"],
    [
      "Trabajadores de los servicios de restauracion, personales, proteccion y vendedores",
      "Servicios y ventas",
    ],
    [
      "Trabajadores cualificados en el sector agricola, ganadero, forestal y pesquero",
      "Agrario cualificado",
    ],
    [
      "Artesanos y trabajadores cualificados de las industrias manufactureras y la construccion (excepto operadores de instalaciones y maquinaria)",
      "Industria/construccion",
    ],
    ["Operadores de instalaciones y maquinaria, y montadores", "Operadores"],
    ["Ocupaciones elementales", "Elementales"],
    ["Ocupaciones militares", "Militares"],
  ];
  const match = labels.find(([full]) => normalizeText(value) === normalizeText(full));
  return match ? match[1] : shortLabel(value, 18);
}

function render() {
  const periodRows = getFilteredPeriodRows();
  const trendRows = getTrendRows();
  document.getElementById("latestPeriod").textContent = state.selectedPeriod;
  renderKpis(periodRows);
  renderFilterSummary(periodRows, trendRows);
  renderCategoryBreakdown(periodRows);
  renderEmploymentBar(periodRows);
  renderExposureRanking(periodRows);
  renderScatter(periodRows);
  renderTrend(trendRows);
  renderTable(periodRows);
}

async function init() {
  try {
    let text = "";
    let loadedFrom = "";

    for (const path of DATA_PATHS) {
      try {
        const response = await fetch(path);
        if (!response.ok) continue;
        text = await response.text();
        loadedFrom = path;
        break;
      } catch (error) {
        console.warn(`No se pudo cargar ${path}`, error);
      }
    }

    if (!text) {
      throw new Error(
        "No se pudo cargar el CSV. Abre el dashboard desde un servidor local en la raiz del proyecto, por ejemplo: python -m http.server 8000"
      );
    }

    state.rows = cleanRows(parseCsv(text));
    state.periods = uniqueSorted(state.rows.map((row) => row.periodo), true);
    state.years = uniqueSorted(state.rows.map((row) => row.year)).filter(Boolean);
    state.selectedPeriod = state.periods[0];
    state.selectedStartYear = state.years.includes(2021) ? 2021 : state.years[0];
    document.getElementById("dataStatus").textContent = `Dataset cargado (${loadedFrom})`;
    setupControls();
    render();
  } catch (error) {
    document.getElementById("dataStatus").textContent =
      "No se pudo cargar el CSV. Sirve la raiz del proyecto en http://localhost:8000/dashboard/";
    console.error(error);
  }
}

init();
