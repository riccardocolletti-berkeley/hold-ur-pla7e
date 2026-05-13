// Canvas drawing controller: captures pointer events, paints the path,
// and exposes the raw point list for upload.

class DrawingCanvas {
  constructor(canvasEl, hintEl, aspectRatio = 1, safeInsetFrac = 0.1) {
    this.el = canvasEl;
    this.hint = hintEl;
    this.ctx = canvasEl.getContext("2d");
    this.points = [];        // [[x_px, y_px], ...] in CSS pixels
    this.drawing = false;
    this.closed = false;     // when true, draws the last->first bridge
    this.aspect = aspectRatio; // width / height, derived from plate_size
    this.safeInset = safeInsetFrac;
    this.width = 600;
    this.height = 600;

    this._fitToScreen();
    window.addEventListener("resize", () => this._fitToScreen());

    const start = (e) => this._start(this._pos(e), e);
    const move  = (e) => this._move(this._pos(e), e);
    const end   = ()  => this._end();

    canvasEl.addEventListener("pointerdown", start);
    canvasEl.addEventListener("pointermove", move);
    canvasEl.addEventListener("pointerup", end);
    canvasEl.addEventListener("pointercancel", end);
    canvasEl.addEventListener("pointerleave", end);
  }

  _fitToScreen() {
    // Pick the largest rectangle that respects plate aspect and fits the viewport.
    const maxByWidth = Math.min(window.innerWidth - 48, 720);
    const maxByHeight = Math.max(220, window.innerHeight - 340);
    let w = maxByWidth;
    let h = w / this.aspect;
    if (h > maxByHeight) {
      h = maxByHeight;
      w = h * this.aspect;
    }
    this.width = Math.floor(w);
    this.height = Math.floor(h);

    // Backing store at native resolution for crisp strokes on retina / iPad.
    const dpr = window.devicePixelRatio || 1;
    this.el.style.width = this.width + "px";
    this.el.style.height = this.height + "px";
    this.el.width = this.width * dpr;
    this.el.height = this.height * dpr;
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this._redraw();
  }

  _pos(e) {
    // Clamp pointer to the safe inner rectangle so the recorded path matches
    // what the simulator will execute.
    const rect = this.el.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const ix = this.width  * this.safeInset;
    const iy = this.height * this.safeInset;
    return [
      Math.min(this.width - ix, Math.max(ix, x)),
      Math.min(this.height - iy, Math.max(iy, y)),
    ];
  }

  _start(p, e) {
    e.preventDefault();
    this.el.setPointerCapture?.(e.pointerId);
    this.points = [p];
    this.drawing = true;
    this.hint?.classList.add("hidden");
    this._redraw();
  }

  _move(p, e) {
    if (!this.drawing) return;
    e.preventDefault();
    const last = this.points[this.points.length - 1];
    // Drop near-duplicate samples to keep the path size sane.
    if (Math.hypot(p[0] - last[0], p[1] - last[1]) >= 1.5) {
      this.points.push(p);
      this._drawSegment(last, p);
    }
  }

  _end() {
    this.drawing = false;
    this._redraw();
  }

  setClosed(flag) {
    this.closed = !!flag;
    this._redraw();
  }

  _drawSegment(a, b) {
    const ctx = this.ctx;
    ctx.lineWidth = 3;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.strokeStyle = "#FDB515";
    ctx.shadowColor = "rgba(253,181,21,0.55)";
    ctx.shadowBlur = 8;
    ctx.beginPath();
    ctx.moveTo(a[0], a[1]);
    ctx.lineTo(b[0], b[1]);
    ctx.stroke();
    ctx.shadowBlur = 0;
  }

  _redraw() {
    const w = this.width, h = this.height;
    const ctx = this.ctx;
    ctx.clearRect(0, 0, w, h);

    // Background grid; spacing scales with the shorter side.
    ctx.strokeStyle = "rgba(255,255,255,0.05)";
    ctx.lineWidth = 1;
    const step = Math.min(w, h) / 10;
    for (let x = step; x < w; x += step) {
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
    }
    for (let y = step; y < h; y += step) {
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
    }

    // Safe rectangle: drawing is clamped here so it never reaches the plate edge.
    const ix = w * this.safeInset;
    const iy = h * this.safeInset;
    ctx.save();
    ctx.strokeStyle = "rgba(253,181,21,0.45)";
    ctx.lineWidth = 1.5;
    ctx.setLineDash([5, 5]);
    ctx.strokeRect(ix, iy, w - 2 * ix, h - 2 * iy);
    ctx.restore();

    // Centre cross marks the plate origin.
    ctx.strokeStyle = "rgba(253,181,21,0.25)";
    ctx.beginPath();
    ctx.moveTo(w / 2, h / 2 - 12); ctx.lineTo(w / 2, h / 2 + 12);
    ctx.moveTo(w / 2 - 12, h / 2); ctx.lineTo(w / 2 + 12, h / 2);
    ctx.stroke();

    // User path.
    if (this.points.length > 1) {
      for (let i = 1; i < this.points.length; i++) {
        this._drawSegment(this.points[i - 1], this.points[i]);
      }
    }

    // Dashed bridge B->A, drawn only after the user releases in closed mode.
    if (this.closed && !this.drawing && this.points.length >= 2) {
      const a = this.points[this.points.length - 1];
      const b = this.points[0];
      ctx.save();
      ctx.lineWidth = 2;
      ctx.lineCap = "round";
      ctx.setLineDash([6, 6]);
      ctx.strokeStyle = "rgba(253,181,21,0.55)";
      ctx.shadowColor = "rgba(253,181,21,0.4)";
      ctx.shadowBlur = 6;
      ctx.beginPath();
      ctx.moveTo(a[0], a[1]);
      ctx.lineTo(b[0], b[1]);
      ctx.stroke();
      ctx.restore();
    }
  }

  clear() {
    this.points = [];
    this.hint?.classList.remove("hidden");
    this._redraw();
  }

  hasPath() { return this.points.length >= 2; }

  export() {
    return { points: this.points.slice(), canvas_size: [this.width, this.height] };
  }
}
