// Gallery view: built-in shapes (top) + user drawings (below).
// Built-ins are read-only and have a tunable radius; drawings are deletable.

const Gallery = {
  el: null,
  emptyEl: null,

  init(listEl, emptyEl) {
    this.el = listEl;
    this.emptyEl = emptyEl;
  },

  async refresh() {
    const [presetsRes, listRes] = await Promise.all([
      API.presets().catch(() => ({ items: [] })),
      API.list().catch(() => ({ items: [] })),
    ]);
    const presets = presetsRes.items || [];
    const drawings = listRes.items || [];

    this.el.innerHTML = "";
    this.emptyEl.classList.toggle("visible", presets.length === 0 && drawings.length === 0);

    if (presets.length) {
      this.el.appendChild(this._sectionHeader("Built-in"));
      for (const it of presets) this.el.appendChild(this._presetCard(it));
    }
    if (drawings.length) {
      this.el.appendChild(this._sectionHeader("Your Drawings"));
      for (const it of drawings) this.el.appendChild(this._drawingCard(it));
    }
  },

  _sectionHeader(text) {
    const h = document.createElement("h3");
    h.className = "gallery-section";
    h.textContent = text;
    return h;
  },

  // ---------- card builders ----------

  _presetCard(item) {
    const card = document.createElement("div");
    card.className = "gallery-card preset";
    card.appendChild(this._badge("Built-in"));

    const thumb = this._thumbnail(item.preview_xy || []);
    card.appendChild(thumb);

    const meta = document.createElement("div");
    meta.className = "gallery-meta";
    card.appendChild(meta);

    // The control row(s) and the run-time params depend on the shape:
    // stationary takes (x, y); circle / figure8 take a single radius.
    const ctrlSel = document.createElement("select");
    ctrlSel.className = "ctrl-select";
    App._populateControllers(ctrlSel);

    const actions = document.createElement("div");
    actions.className = "gallery-actions";
    const runBtn = document.createElement("button");
    runBtn.className = "btn gold";
    runBtn.textContent = "Run";
    actions.append(runBtn);

    if (item.shape === "stationary") {
      meta.innerHTML = `
        <span class="gallery-name">${escapeHtml(item.label)}</span>
        <span class="gallery-sub" data-role="hold-label">target (${(item.x * 100).toFixed(1)}, ${(item.y * 100).toFixed(1)}) cm</span>
      `;
      const xRow = this._positionSlider("X", item.x_bounds, item.x);
      const yRow = this._positionSlider("Y", item.y_bounds, item.y);
      card.append(xRow.row, yRow.row);
      const label = meta.querySelector('[data-role="hold-label"]');
      const updateLabel = () => {
        label.textContent =
          `target (${(xRow.input.valueAsNumber * 100).toFixed(1)}, ${(yRow.input.valueAsNumber * 100).toFixed(1)}) cm`;
        this._setThumbnailPoint(thumb, xRow.input.valueAsNumber,
                                 yRow.input.valueAsNumber, item.x_bounds, item.y_bounds);
      };
      xRow.input.addEventListener("input", updateLabel);
      yRow.input.addEventListener("input", updateLabel);
      updateLabel();
      runBtn.onclick = async () => {
        runBtn.disabled = true;
        const r = await API.runPreset(item.shape, {
          x: xRow.input.valueAsNumber,
          y: yRow.input.valueAsNumber,
        }, ctrlSel.value);
        runBtn.disabled = false;
        App._handleRunResponse(r, item.label);
      };
    } else {
      meta.innerHTML = `
        <span class="gallery-name">${escapeHtml(item.label)}</span>
        <span class="gallery-sub" data-role="radius-label">radius ${(item.radius * 100).toFixed(1)} cm</span>
      `;
      const slider = document.createElement("input");
      slider.type = "range";
      slider.className = "preset-slider";
      slider.min = item.min_radius;
      slider.max = item.max_radius;
      slider.step = 0.005;
      slider.value = item.radius;
      bindSliderFill(slider);
      const radLabel = meta.querySelector('[data-role="radius-label"]');
      slider.addEventListener("input", () => {
        const r = parseFloat(slider.value);
        radLabel.textContent = `radius ${(r * 100).toFixed(1)} cm`;
        this._scaleThumbnail(thumb, r / item.radius);
      });
      card.appendChild(slider);
      runBtn.onclick = async () => {
        runBtn.disabled = true;
        const r = await API.runPreset(item.shape, {
          radius: parseFloat(slider.value),
        }, ctrlSel.value);
        runBtn.disabled = false;
        App._handleRunResponse(r, item.label);
      };
    }

    card.appendChild(ctrlSel);
    card.appendChild(actions);
    return card;
  },

  _positionSlider(axis, bounds, value) {
    const row = document.createElement("div");
    row.className = "position-row";
    const lo = bounds?.[0] ?? -0.1;
    const hi = bounds?.[1] ?? 0.1;
    row.innerHTML = `
      <span class="position-label">${axis}</span>
      <input type="range" class="preset-slider" min="${lo}" max="${hi}" step="0.005" value="${value}">
    `;
    const input = row.querySelector("input");
    bindSliderFill(input);
    return { row, input };
  },

  _setThumbnailPoint(svg, x, y, x_bounds, y_bounds) {
    // Re-draw the single dot at the chosen target, scaled to the SVG's viewport.
    const path = svg.querySelector("path");
    if (!path) return;
    const W = 200, H = 140, pad = 10;
    const sx = (W - 2 * pad) / Math.max(1e-6, x_bounds[1] - x_bounds[0]);
    const sy = (H - 2 * pad) / Math.max(1e-6, y_bounds[1] - y_bounds[0]);
    const px = W / 2 + x * sx;
    const py = H / 2 - y * sy;
    path.setAttribute("d", `M${px - 3},${py} a3,3 0 1,0 6,0 a3,3 0 1,0 -6,0`);
    path.setAttribute("transform", "");
  },

  _drawingCard(item) {
    const card = document.createElement("div");
    card.className = "gallery-card";

    card.appendChild(this._thumbnail(item.preview_xy || []));

    const meta = document.createElement("div");
    meta.className = "gallery-meta";
    meta.innerHTML = `
      <span class="gallery-name">${escapeHtml(item.name)}</span>
      <span class="gallery-sub">${item.duration_s}s · ${item.path_mode} · ${item.end_mode}</span>
    `;
    card.appendChild(meta);

    const ctrlSel = document.createElement("select");
    ctrlSel.className = "ctrl-select";
    App._populateControllers(ctrlSel);
    card.appendChild(ctrlSel);

    const actions = document.createElement("div");
    actions.className = "gallery-actions";

    const runBtn = document.createElement("button");
    runBtn.className = "btn gold";
    runBtn.textContent = "Run";
    runBtn.onclick = async () => {
      runBtn.disabled = true;
      const r = await API.run(item.name, ctrlSel.value);
      runBtn.disabled = false;
      App._handleRunResponse(r, item.name);
    };

    const delBtn = document.createElement("button");
    delBtn.className = "btn danger";
    delBtn.textContent = "Delete";
    delBtn.onclick = async () => {
      if (!confirm(`Delete "${item.name}"?`)) return;
      await API.remove(item.name);
      this.refresh();
    };

    actions.append(runBtn, delBtn);
    card.appendChild(actions);
    return card;
  },

  _badge(text) {
    const b = document.createElement("span");
    b.className = "card-tag absolute";
    b.textContent = text;
    return b;
  },

  // ---------- thumbnail ----------

  _thumbnail(points) {
    // Centered (200x140) SVG of the path; flip y for screen coords.
    const W = 200, H = 140, pad = 10;
    const svgNS = "http://www.w3.org/2000/svg";
    const svg = document.createElementNS(svgNS, "svg");
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    svg.dataset.viewBox = `${W} ${H} ${pad}`;

    if (!points.length) return svg;

    let xmin = Infinity, xmax = -Infinity, ymin = Infinity, ymax = -Infinity;
    for (const [x, y] of points) {
      if (x < xmin) xmin = x; if (x > xmax) xmax = x;
      if (y < ymin) ymin = y; if (y > ymax) ymax = y;
    }
    const sx = (W - 2 * pad) / Math.max(1e-6, xmax - xmin);
    const sy = (H - 2 * pad) / Math.max(1e-6, ymax - ymin);
    const s = Math.min(sx, sy);
    const cx = (xmin + xmax) / 2;
    const cy = (ymin + ymax) / 2;

    let d = "";
    points.forEach(([x, y], i) => {
      const px = W / 2 + (x - cx) * s;
      const py = H / 2 - (y - cy) * s;
      d += (i === 0 ? "M" : "L") + px.toFixed(1) + "," + py.toFixed(1);
    });
    const path = document.createElementNS(svgNS, "path");
    path.setAttribute("d", d);
    path.setAttribute("fill", "none");
    path.setAttribute("stroke", "#FDB515");
    path.setAttribute("stroke-width", "2");
    path.setAttribute("stroke-linecap", "round");
    path.setAttribute("stroke-linejoin", "round");
    path.dataset.baseScale = "1";
    svg.appendChild(path);
    return svg;
  },

  _scaleThumbnail(svg, factor) {
    // Live-resize the path around the SVG center for radius-slider feedback.
    const path = svg.querySelector("path");
    if (!path) return;
    path.setAttribute("transform", `translate(100,70) scale(${factor}) translate(-100,-70)`);
  },
};

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));
}

// Drive a range input's gold mask via `--fill`. Inline-styled on every
// `input` event and once at creation, so the fill stays in lock-step with
// the thumb regardless of how the slider was instantiated.
function bindSliderFill(input) {
  const update = () => {
    const min = parseFloat(input.min);
    const max = parseFloat(input.max);
    const val = parseFloat(input.value);
    const pct = max > min ? ((val - min) / (max - min)) * 100 : 0;
    input.style.setProperty("--fill", `${pct}%`);
  };
  input.addEventListener("input", update);
  update();
}
