// Evidence timeline: where the indexes found support for the query, which regions were
// searched, and the returned moments, aligned with the video's playhead.
import { h, clockTime, preciseTime, pct } from './dom.js';

const CHANNELS = [['video', 'Visual'], ['metadata', 'Scenes'], ['speech', 'Speech']];

export class Timeline {
  constructor(root, { onSeek, onSelectMatch }) {
    this.onSeek = onSeek;
    this.onSelectMatch = onSelectMatch;
    this.data = { duration: 0, evidence: [], candidates: [], matches: [], speech: [] };
    this.current = 0;
    this.selected = -1;
    this.keyboardTime = null;

    this.canvas = h('canvas', { class: 'timeline-canvas' });
    this.markers = h('div', { class: 'timeline-markers' });
    this.cursor = h('div', { class: 'timeline-cursor', hidden: true });
    this.playhead = h('div', { class: 'timeline-playhead' });
    this.tooltip = h('div', { class: 'timeline-tooltip', hidden: true });
    this.surface = h('div', {
      class: 'timeline-surface', tabindex: '0', role: 'slider', 'aria-valuemin': '0',
      'aria-label': 'Timeline. Arrow keys move the cursor, Enter jumps to that time.',
    }, this.canvas, this.markers, this.cursor, this.playhead, this.tooltip);
    this.title = h('span', { class: 'timeline-title' }, 'Timeline');
    this.legend = h('div', { class: 'timeline-legend' });
    this.axis = h('div', { class: 'timeline-axis', 'aria-hidden': 'true' });
    root.replaceChildren(h('div', { class: 'timeline-head' }, this.title, this.legend), this.surface, this.axis);

    this.surface.addEventListener('pointermove', (event) => this.showAt(this.timeAt(event.clientX)));
    this.surface.addEventListener('pointerleave', () => this.hideHover());
    this.surface.addEventListener('click', (event) => {
      if (event.target.closest('.timeline-match')) return;
      const time = this.timeAt(event.clientX);
      if (time != null) this.onSeek(time);
    });
    this.surface.addEventListener('keydown', (event) => this.onKey(event));
    this.surface.addEventListener('blur', () => { this.keyboardTime = null; this.hideHover(); });
    new ResizeObserver(() => this.draw()).observe(this.surface);
  }

  set(data) {
    this.data = { ...this.data, ...data };
    this.selected = -1;
    this.surface.setAttribute('aria-valuemax', String(Math.round(this.data.duration)));
    this.renderMarkers();
    this.renderAxis();
    this.renderLegend();
    this.draw();
  }

  setTime(time) {
    this.current = time;
    const { duration } = this.data;
    this.playhead.style.left = duration ? `${Math.min(100, (time / duration) * 100)}%` : '0';
    this.surface.setAttribute('aria-valuenow', String(Math.round(time)));
    this.surface.setAttribute('aria-valuetext', clockTime(time));
  }

  select(index) {
    this.selected = index;
    for (const marker of this.markers.children) {
      marker.classList.toggle('is-selected', Number(marker.dataset.index) === index);
    }
  }

  timeAt(clientX) {
    const rect = this.surface.getBoundingClientRect();
    if (!this.data.duration || !rect.width) return null;
    return Math.min(1, Math.max(0, (clientX - rect.left) / rect.width)) * this.data.duration;
  }

  renderMarkers() {
    const { duration, matches } = this.data;
    this.markers.replaceChildren(...(duration ? matches.map((match, index) => h('button', {
      type: 'button',
      class: 'timeline-match',
      dataset: { index },
      style: { left: `${(match.start / duration) * 100}%`, width: `${((match.end - match.start) / duration) * 100}%` },
      'aria-label': `Play moment ${index + 1}, ${preciseTime(match.start)} to ${preciseTime(match.end)}`,
      title: `Moment ${index + 1} · ${preciseTime(match.start)} – ${preciseTime(match.end)}`,
      onclick: () => this.onSelectMatch(index),
    }, h('span', { class: 'timeline-match-label' }, index + 1))) : []));
  }

  renderAxis() {
    const { duration } = this.data;
    const ticks = duration ? [0, 0.25, 0.5, 0.75, 1] : [];
    this.axis.replaceChildren(...ticks.map((f) => h('span', {
      class: f === 0 ? 'tick-start' : f === 1 ? 'tick-end' : '',
      style: { left: `${f * 100}%` },
    }, clockTime(f * duration))));
  }

  renderLegend() {
    const { evidence, candidates, matches, speech } = this.data;
    const item = (swatch, label) => h('span', { class: 'legend-item' }, h('span', { class: `legend-swatch swatch-${swatch}` }), label);
    const items = [];
    if (evidence.length) items.push(item('evidence', 'Evidence for your query'));
    if (candidates.length) items.push(item('candidate', 'Searched regions'));
    if (matches.length) items.push(item('match', 'Moments'));
    if (!evidence.length && speech.length) items.push(item('speech', 'Speech'));
    this.legend.replaceChildren(...items);
  }

  draw() {
    const rect = this.surface.getBoundingClientRect();
    const width = Math.max(1, rect.width);
    const height = Math.max(1, rect.height);
    const ratio = window.devicePixelRatio || 1;
    if (this.canvas.width !== Math.round(width * ratio) || this.canvas.height !== Math.round(height * ratio)) {
      this.canvas.width = Math.round(width * ratio);
      this.canvas.height = Math.round(height * ratio);
    }
    const ctx = this.canvas.getContext('2d');
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    ctx.clearRect(0, 0, width, height);

    const style = getComputedStyle(this.surface);
    const color = (name) => style.getPropertyValue(name).trim();
    const { duration, evidence, candidates, speech } = this.data;
    const top = 6;
    const baseline = height - 24;
    const x = (time) => (duration ? (time / duration) * width : 0);

    ctx.fillStyle = color('--tl-grid');
    ctx.fillRect(0, Math.round(baseline), width, 1);
    if (!duration) return;

    ctx.fillStyle = color('--tl-candidate');
    for (const region of candidates) {
      ctx.fillRect(x(region.start), top, Math.max(1, x(region.end) - x(region.start)), baseline - top);
    }

    const peak = evidence.reduce((max, row) => Math.max(max, row.score || 0), 0);
    if (evidence.length && peak > 0) {
      // Heights are relative to the strongest evidence; the tooltip carries raw values.
      const points = evidence.map((row) => [x((row.start + row.end) / 2), baseline - ((row.score || 0) / peak) * (baseline - top - 2)]);
      const trace = () => {
        ctx.moveTo(0, points[0][1]);
        for (const [px, py] of points) ctx.lineTo(px, py);
        ctx.lineTo(width, points[points.length - 1][1]);
      };
      ctx.beginPath();
      ctx.moveTo(0, baseline);
      ctx.lineTo(0, points[0][1]);
      trace();
      ctx.lineTo(width, baseline);
      ctx.closePath();
      ctx.fillStyle = color('--tl-evidence-wash');
      ctx.fill();
      ctx.beginPath();
      trace();
      ctx.lineWidth = 2;
      ctx.lineJoin = 'round';
      ctx.lineCap = 'round';
      ctx.strokeStyle = color('--tl-evidence');
      ctx.stroke();
    } else if (speech.length) {
      ctx.fillStyle = color('--tl-speech');
      for (const segment of speech) {
        ctx.fillRect(x(segment.start), baseline - 9, Math.max(2, x(segment.end) - x(segment.start) - 1), 6);
      }
    }

    // Numbers only go inside moments wide enough to hold them without clipping.
    for (const marker of this.markers.children) marker.classList.toggle('has-label', marker.offsetWidth >= 20);
  }

  nearestEvidence(time) {
    const rows = this.data.evidence;
    let low = 0;
    let high = rows.length - 1;
    while (low <= high) {
      const mid = (low + high) >> 1;
      if (rows[mid].end < time) low = mid + 1;
      else if (rows[mid].start > time) high = mid - 1;
      else return rows[mid];
    }
    return rows[Math.min(rows.length - 1, Math.max(0, low))] || null;
  }

  showAt(time) {
    if (time == null) return;
    const width = this.surface.getBoundingClientRect().width;
    const left = (time / this.data.duration) * width;
    this.cursor.hidden = false;
    this.cursor.style.left = `${left}px`;

    const row = (value, label, strong = false) => h('div', { class: `tt-row${strong ? ' tt-strong' : ''}` },
      h('span', { class: 'tt-value' }, value), h('span', { class: 'tt-label' }, label));
    const lines = [h('div', { class: 'tt-time' }, preciseTime(time))];
    const evidence = this.nearestEvidence(time);
    if (evidence) {
      const channels = evidence.channel_scores || {};
      const values = { ...channels, speech: Math.max(channels.transcript_semantic || 0, channels.transcript_bm25 || 0) };
      lines.push(row(pct(evidence.score), 'Evidence', true));
      for (const [key, label] of CHANNELS) lines.push(row(pct(values[key]), label));
    }
    const match = this.data.matches.findIndex((m) => time >= m.start && time <= m.end);
    if (match >= 0) lines.push(h('div', { class: 'tt-match' }, h('span', { class: 'legend-swatch swatch-match' }), `Moment ${match + 1}`));
    if (lines.length === 1) lines.push(h('div', { class: 'tt-hint' }, 'Click to jump here'));

    this.tooltip.replaceChildren(...lines);
    this.tooltip.hidden = false;
    const tipWidth = this.tooltip.offsetWidth;
    this.tooltip.style.left = `${Math.min(Math.max(0, left - tipWidth / 2), Math.max(0, width - tipWidth))}px`;
  }

  hideHover() {
    this.cursor.hidden = true;
    this.tooltip.hidden = true;
  }

  onKey(event) {
    const { duration } = this.data;
    if (!duration) return;
    const step = (event.shiftKey ? 0.1 : 0.02) * duration;
    let time = this.keyboardTime ?? this.current;
    if (event.key === 'ArrowRight') time += step;
    else if (event.key === 'ArrowLeft') time -= step;
    else if (event.key === 'Home') time = 0;
    else if (event.key === 'End') time = duration;
    else if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      this.onSeek(time);
      return;
    } else return;
    event.preventDefault();
    this.keyboardTime = Math.min(duration, Math.max(0, time));
    this.showAt(this.keyboardTime);
  }
}
