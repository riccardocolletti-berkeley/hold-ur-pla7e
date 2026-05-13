// SPA controller: view switching, splash, draw wiring, backend toggle,
// real-backend launch sheet, and idle status polling.

const App = {
  view: null,
  pathMode: "open",
  canvas: null,
  cfg: null,
  backend: "sim",

  async init() {
    this.cfg = await API.config().catch(() => null);
    this.backend = this.cfg?.current_backend || "sim";

    this._wireBackendToggle();
    this._populateControllers(document.getElementById("controller"));
    setTimeout(() => this.show("mode"), 1400);

    // Mode picker.
    document.querySelectorAll(".mode-card").forEach((c) => {
      c.addEventListener("click", () => {
        this.pathMode = c.dataset.mode;
        document.getElementById("draw-mode-label").textContent =
          this.pathMode === "closed" ? "Closed Loop" : "Open Path";
        document.getElementById("end-mode-row").style.display =
          this.pathMode === "closed" ? "none" : "";
        this.canvas?.setClosed(this.pathMode === "closed");
        this.show("draw");
      });
      c.addEventListener("pointermove", (e) => {
        const r = c.getBoundingClientRect();
        c.style.setProperty("--mx", `${((e.clientX - r.left) / r.width) * 100}%`);
        c.style.setProperty("--my", `${((e.clientY - r.top) / r.height) * 100}%`);
      });
    });

    // Bottom nav + back arrows.
    document.querySelectorAll("[data-view]").forEach((b) =>
      b.addEventListener("click", () => this.show(b.dataset.view))
    );
    document.querySelectorAll("[data-back]").forEach((b) =>
      b.addEventListener("click", () => this.show(b.dataset.back))
    );

    // Drawing canvas; aspect mirrors the plate, safe inset matches the server's mapping.
    const [px, py] = this.cfg?.plate_size || [1, 1];
    this.canvas = new DrawingCanvas(
      document.getElementById("canvas"),
      document.getElementById("canvas-hint"),
      px / py,
      this.cfg?.safe_inset_frac ?? 0.1,
    );

    // Duration slider.
    const durIn = document.getElementById("duration");
    const durOut = document.getElementById("dur-val");
    const updateSlider = () => {
      durOut.textContent = durIn.value;
      const pct = ((durIn.value - durIn.min) / (durIn.max - durIn.min)) * 100;
      durIn.style.setProperty("--fill", pct + "%");
    };
    durIn.addEventListener("input", updateSlider);
    updateSlider();

    // Action buttons.
    document.getElementById("clear").addEventListener("click", () => this.canvas.clear());
    document.getElementById("save").addEventListener("click", () => this.save(false));
    document.getElementById("save-run").addEventListener("click", () => this.save(true));
    document.getElementById("stop-sim").addEventListener("click", async () => {
      await API.stop();
      this.toast("Stopped");
      this._refreshStatus();
    });

    // Launch sheet (real backend) close + copy.
    document.getElementById("launch-close").addEventListener("click", () => {
      document.getElementById("launch-sheet").classList.remove("visible");
    });
    document.getElementById("launch-copy").addEventListener("click", () => {
      const cmd = document.getElementById("launch-cmd").textContent;
      navigator.clipboard?.writeText(cmd);
      this.toast("Command copied");
    });

    Gallery.init(
      document.getElementById("gallery-list"),
      document.getElementById("gallery-empty"),
    );
    Training.init();

    this._refreshStats();
    this._refreshStatus();
    setInterval(() => this._refreshStatus(), 3000);
  },

  show(name) {
    document.querySelectorAll(".screen").forEach((s) => s.classList.remove("active"));
    document.getElementById(name)?.classList.add("active");
    this.view = name;
    document.querySelectorAll(".nav-btn").forEach((b) =>
      b.classList.toggle("active", b.dataset.view === name)
    );
    if (name === "gallery") {
      Gallery.refresh();
    } else if (name === "mode") {
      this._refreshStats();
    } else if (name === "training") {
      Training.onShow();
    }
    if (name !== "training") Training.onHide();
  },

  async save(runAfter) {
    if (!this.canvas.hasPath()) {
      this.toast("Draw something first", true);
      return;
    }
    const exp = this.canvas.export();
    const payload = {
      points: exp.points,
      canvas_size: exp.canvas_size,
      duration_s: parseFloat(document.getElementById("duration").value),
      path_mode: this.pathMode,
      end_mode: this.pathMode === "closed"
        ? "loop"
        : document.getElementById("end-mode").value,
      name: document.getElementById("name").value,
    };
    const res = await API.save(payload);
    if (!res.ok) {
      this.toast(res.error || "Save failed", true);
      return;
    }
    this.toast(`Saved as ${res.name}`);
    document.getElementById("name").value = "";
    if (runAfter) {
      const ctrl = document.getElementById("controller").value;
      const r = await API.run(res.name, ctrl);
      this._handleRunResponse(r, res.name);
      this._refreshStatus();
    }
  },

  _wireBackendToggle() {
    const toggle = document.getElementById("backend-toggle");
    if (!toggle) return;
    // Mark the current backend as selected on first paint.
    toggle.querySelectorAll(".backend-opt").forEach((b) =>
      b.classList.toggle("active", b.dataset.backend === this.backend)
    );
    toggle.addEventListener("click", async (e) => {
      const btn = e.target.closest(".backend-opt");
      if (!btn) return;
      const next = btn.dataset.backend;
      if (next === this.backend) return;
      const res = await API.setBackend(next).catch(() => null);
      if (!res?.ok) {
        this.toast("Backend switch failed", true);
        return;
      }
      this.backend = next;
      toggle.querySelectorAll(".backend-opt").forEach((b) =>
        b.classList.toggle("active", b.dataset.backend === this.backend)
      );
      this._populateControllers(document.getElementById("controller"));
      // Gallery cards each carry their own controller dropdown, populated at
      // render time. Re-rendering the visible gallery is what propagates the
      // new backend's filter into the cards already on screen.
      if (this.view === "gallery") Gallery.refresh();
      this.toast(`Backend: ${this.backend === "sim" ? "Simulator" : "Real Robot"}`);
    });
  },

  _populateControllers(sel) {
    if (!sel || !this.cfg?.controllers) return;
    sel.innerHTML = "";
    // Filter to controllers that the active backend supports.
    for (const c of this.cfg.controllers) {
      if (Array.isArray(c.available_for) && !c.available_for.includes(this.backend)) {
        continue;
      }
      const opt = document.createElement("option");
      opt.value = c.id;
      opt.textContent = c.label;
      sel.appendChild(opt);
    }
  },

  _handleRunResponse(r, label) {
    if (!r?.ok) {
      this.toast(`Failed to start ${label}`, true);
      return;
    }
    if (r.backend === "real" && r.command) {
      this._showLaunchSheet(r.command, r.staged_file);
      return;
    }
    this.toast(`Running ${label}`);
  },

  _showLaunchSheet(command, staged) {
    document.getElementById("launch-cmd").textContent = command;
    const stagedEl = document.getElementById("launch-staged");
    if (staged) {
      stagedEl.textContent = `Drawing staged at ${staged}`;
      stagedEl.style.display = "";
    } else {
      stagedEl.style.display = "none";
    }
    document.getElementById("launch-sheet").classList.add("visible");
  },

  async _refreshStats() {
    const list = await API.list().catch(() => null);
    if (list) document.getElementById("stat-count").textContent = list.items?.length ?? 0;
    if (this.cfg?.plate_size) {
      const [px, py] = this.cfg.plate_size;
      document.getElementById("stat-plate").textContent =
        `${(px * 100).toFixed(0)}×${(py * 100).toFixed(0)}`;
    }
    if (this.cfg?.ball_radius != null) {
      document.getElementById("stat-ball").textContent =
        (this.cfg.ball_radius * 2000).toFixed(0);
    }
  },

  async _refreshStatus() {
    const s = await API.status().catch(() => null);
    const pill = document.getElementById("status-pill");
    if (!pill) return;
    const running = !!s?.running;
    pill.classList.toggle("running", running);
    const labelMap = { sim: "Sim", real: "Real" };
    const tag = labelMap[s?.backend || this.backend] || "-";
    pill.querySelector(".status-text").textContent =
      running ? `${tag} · Running` : `${tag} · Idle`;
  },

  toast(msg, isError) {
    const el = document.getElementById("toast");
    el.textContent = msg;
    el.classList.toggle("error", !!isError);
    el.classList.add("show");
    clearTimeout(this._toastT);
    this._toastT = setTimeout(() => el.classList.remove("show"), 2200);
  },
};

// `gallery.js` and `training.js` rely on `App.toast` / `App._populateControllers`.
// Kept on the window scope so the existing helpers find them.

document.addEventListener("DOMContentLoaded", () => App.init());
