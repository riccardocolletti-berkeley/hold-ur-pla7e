// Live RL training dashboard. Polls /api/training/* and renders Plotly charts.

const Training = {
  selEl:   null,
  pillEl:  null,
  kpiEl:   null,
  refreshT: null,
  currentRun: null,
  // Last step count we displayed; used to detect whether new metrics arrived
  // between two refreshes, which is what "running" actually means here.
  _lastStep: -1,

  CHARTS: [
    { id: "chart-rew",   key: "ep_rew_mean",          color: "#FDB515" },
    { id: "chart-len",   key: "ep_len_mean",          color: "#5dd39e" },
    { id: "chart-vloss", key: "value_loss",           color: "#3aa1d6" },
    { id: "chart-ploss", key: "policy_gradient_loss", color: "#ff8a5b" },
    { id: "chart-kl",    key: "approx_kl",            color: "#c69cff" },
    { id: "chart-fps",   key: "fps",                  color: "#ffffff" },
  ],

  async init() {
    this.selEl  = document.getElementById("run-select");
    this.pillEl = document.getElementById("train-pill");
    this.kpiEl  = document.getElementById("kpi-row");

    this.selEl.addEventListener("change", () => this.loadRun(this.selEl.value));
    document.getElementById("refresh-now").onclick = () => this.refresh();
    document.getElementById("auto-refresh").addEventListener("change", (e) => {
      if (e.target.checked) this._startTimer(); else this._stopTimer();
    });
  },

  async refreshRunList() {
    const { items = [] } = await API.trainingRuns().catch(() => ({ items: [] }));
    if (!items.length) {
      this.selEl.innerHTML = `<option value="">(no runs yet)</option>`;
      this.pillEl.classList.remove("running");
      this.pillEl.querySelector(".status-text").textContent = "No runs";
      return;
    }
    const prev = this.selEl.value;
    this.selEl.innerHTML = items.map((it) => {
      const tag = it.has_final ? " ✓" : "";
      return `<option value="${it.name}">${it.name}${tag}</option>`;
    }).join("");
    if (items.find((i) => i.name === prev)) this.selEl.value = prev;
    if (!this.currentRun) this.loadRun(this.selEl.value);
  },

  async refresh() {
    if (!this.currentRun) {
      await this.refreshRunList();
      return;
    }
    await this.refreshRunList();
    const data = await API.trainingMetrics(this.currentRun).catch(() => null);
    if (!data || !data.steps) return;
    this._renderKPIs(data);
    for (const cfg of this.CHARTS) this._plot(cfg, data);

    // Liveness: pill turns gold only when the step count grew since the
    // previous refresh; a static run keeps the idle styling.
    const last = data.steps[data.steps.length - 1] ?? 0;
    const grew = last > this._lastStep;
    this._lastStep = last;
    this.pillEl.querySelector(".status-text").textContent =
      `${(last / 1000).toFixed(0)}k steps`;
    this.pillEl.classList.toggle("running", grew);
  },

  loadRun(name) {
    if (!name) return;
    this.currentRun = name;
    this._lastStep = -1;
    return this.refresh();
  },

  _renderKPIs(data) {
    const last = (k) => {
      const arr = data.series[k];
      return arr && arr.length ? arr[arr.length - 1] : null;
    };
    const fmt = (v, d = 2) => (v == null ? "-" : Number(v).toFixed(d));
    const cells = [
      { lbl: "Steps",       val: fmt(data.steps.at(-1) / 1000, 0) + "k" },
      { lbl: "Reward",      val: fmt(last("ep_rew_mean")) },
      { lbl: "Ep length",   val: fmt(last("ep_len_mean"), 0) },
      { lbl: "Value loss",  val: fmt(last("value_loss"), 3) },
      { lbl: "FPS",         val: fmt(last("fps"), 0) },
    ];
    this.kpiEl.innerHTML = cells.map(c =>
      `<div class="kpi"><span class="kpi-val">${c.val}</span><span class="kpi-lbl">${c.lbl}</span></div>`
    ).join("");
  },

  _plot({ id, key, color }, data) {
    const ys = data.series[key];
    if (!ys) return;
    const trace = {
      x: data.steps,
      y: ys,
      mode: "lines",
      line: { color, width: 2 },
      hovertemplate: "%{y:.4f}<extra>step %{x}</extra>",
    };
    const layout = {
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor:  "rgba(0,0,0,0)",
      font: { color: "#98a8be", family: "system-ui, -apple-system, sans-serif", size: 11 },
      margin: { l: 44, r: 12, t: 8, b: 32 },
      xaxis: { gridcolor: "rgba(255,255,255,0.05)", zerolinecolor: "rgba(255,255,255,0.1)" },
      yaxis: { gridcolor: "rgba(255,255,255,0.05)", zerolinecolor: "rgba(255,255,255,0.1)" },
      showlegend: false,
    };
    Plotly.react(id, [trace], layout, { displayModeBar: false, responsive: true });
  },

  _startTimer() {
    this._stopTimer();
    this.refreshT = setInterval(() => this.refresh(), 5000);
  },
  _stopTimer() { if (this.refreshT) { clearInterval(this.refreshT); this.refreshT = null; } },

  onShow() { this.refresh(); this._startTimer(); },
  onHide() { this._stopTimer(); },
};
