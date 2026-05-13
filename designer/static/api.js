// Thin fetch wrappers around the Flask API.

const API = {
  async config() {
    const r = await fetch("/api/config");
    return r.json();
  },
  async setBackend(backend) {
    const r = await fetch("/api/backend", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ backend }),
    });
    return r.json();
  },
  async save(payload) {
    const r = await fetch("/api/save", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    return r.json();
  },
  async list() {
    const r = await fetch("/api/list");
    return r.json();
  },
  async remove(name) {
    const r = await fetch(`/api/drawing/${encodeURIComponent(name)}`, { method: "DELETE" });
    return r.json();
  },
  async run(name, controller) {
    const r = await fetch("/api/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name, controller }),
    });
    return r.json();
  },
  async presets() {
    const r = await fetch("/api/presets");
    return r.json();
  },
  async runPreset(shape, params, controller) {
    const body = { shape, controller, ...(params || {}) };
    const r = await fetch("/api/run_preset", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    return r.json();
  },
  async stop() {
    const r = await fetch("/api/stop", { method: "POST" });
    return r.json();
  },
  async status() {
    const r = await fetch("/api/status");
    return r.json();
  },
  async trainingRuns() {
    const r = await fetch("/api/training/runs");
    return r.json();
  },
  async trainingMetrics(name) {
    const r = await fetch(`/api/training/metrics/${encodeURIComponent(name)}`);
    return r.json();
  },
};
