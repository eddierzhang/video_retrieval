import { api, uploadVideo } from './api.js';
import { h, icon, hydrateIcons, clockTime, preciseTime, formatElapsed, formatBytes, relativeTime, pct, highlight } from './dom.js';
import { Timeline } from './timeline.js';

const ACTIVE = ['queued', 'running'];
const EXAMPLES = ['a person walking through a doorway', 'someone says “thank you”', 'read the text on the sign'];
const CHANNEL_LABELS = {
  video: 'Visual frames', metadata: 'Scene descriptions', transcript_semantic: 'Speech meaning', transcript_bm25: 'Speech keywords',
};

const storage = {
  get(key) { try { return localStorage.getItem(key); } catch { return null; } },
  set(key, value) { try { localStorage.setItem(key, value); } catch { /* private mode */ } },
};

const plural = (count, word) => `${count} ${word}${count === 1 ? '' : 's'}`;
const byId = (id) => document.getElementById(id);
const els = {
  library: byId('library'), libraryCount: byId('library-count'), addVideo: byId('add-video'), fileInput: byId('file-input'),
  systemPill: byId('system-pill'), banner: byId('banner'), empty: byId('empty-state'), emptyChoose: byId('empty-choose'),
  workspace: byId('workspace'), title: byId('video-title'), meta: byId('video-meta'), actions: byId('video-actions'),
  form: byId('search-form'), input: byId('search-input'), submit: byId('search-submit'), options: byId('search-options'),
  modeButtons: [...document.querySelectorAll('[data-mode]')], optRefine: byId('opt-refine'),
  optConfidence: byId('opt-confidence'), optConfidenceOut: byId('opt-confidence-out'),
  optCandidates: byId('opt-candidates'), optCandidatesOut: byId('opt-candidates-out'),
  video: byId('video'), overlay: byId('player-overlay'), timeline: byId('timeline'),
  tabs: [...document.querySelectorAll('[role="tab"]')], tabPanel: byId('tab-panel'), results: byId('results'),
  dropOverlay: byId('drop-overlay'), settings: byId('settings-dialog'), confirm: byId('confirm-dialog'), toasts: byId('toasts'),
};

const state = {
  system: null,
  offline: false,
  videos: [],
  uploads: [],
  selectedId: null,
  video: null,
  transcript: [],
  transcriptFilter: '',
  searchId: null,
  search: null,
  tab: 'plan',
  mode: storage.get('moments.mode') === 'quick' ? 'quick' : 'verified',
  activeMatch: -1,
  segmentEnd: null,
  overlayDismissed: new Set(),
};

// Render functions skip work when their inputs are unchanged, which keeps open <details>,
// scroll positions, and hover states intact across polling updates.
const rendered = {};
function changed(name, key) {
  const serialized = JSON.stringify(key);
  if (rendered[name] === serialized) return false;
  rendered[name] = serialized;
  return true;
}

const timeline = new Timeline(els.timeline, { onSeek: (time) => seek(time), onSelectMatch: (index) => playMatch(index) });
let selectToken = 0;
let pollTimer = null;
let programmaticSeek = false;
let lastTranscriptIndex = -1;

// ------------------------------------------------------------------ data

async function loadSystem() {
  try {
    state.system = await api.system();
  } catch {
    state.system = null;
  }
  renderSystem();
  renderBanner();
}

async function loadVideos() {
  state.videos = (await api.videos()).videos;
  state.offline = false;
}

async function loadSearch(searchId) {
  state.searchId = searchId;
  state.search = null;
  if (!searchId) return;
  try {
    state.search = await api.search(state.selectedId, searchId);
  } catch {
    state.searchId = null;
  }
}

async function loadTranscript() {
  state.transcript = state.video?.index ? (await api.transcript(state.video.id)).segments : [];
  lastTranscriptIndex = -1;
}

async function selectVideo(id, searchId) {
  const token = ++selectToken;
  state.activeMatch = -1;
  state.segmentEnd = null;
  state.transcriptFilter = '';
  if (!id) {
    Object.assign(state, { selectedId: null, video: null, search: null, searchId: null, transcript: [] });
    updateHash();
    render();
    return;
  }
  state.selectedId = id;
  renderLibrary();
  try {
    const video = await api.video(id);
    if (token !== selectToken) return;
    state.video = video;
    const fallback = video.searches.find((s) => s.status === 'done')?.id ?? video.searches[0]?.id ?? null;
    await Promise.all([loadSearch(searchId ?? fallback), loadTranscript()]);
  } catch (error) {
    toast(error.message, 'error');
    return;
  }
  if (token !== selectToken) return;
  updateHash();
  render();
  seekToFirstMatch();
}

function busy() {
  return state.uploads.length > 0 || state.videos.some((v) => v.jobs.length) || ACTIVE.includes(state.search?.status);
}

function schedulePoll(delay) {
  clearTimeout(pollTimer);
  pollTimer = setTimeout(poll, delay ?? (busy() ? 1000 : 5000));
}

async function poll() {
  try {
    await loadVideos();
    if (state.selectedId && !state.videos.some((v) => v.id === state.selectedId)) {
      await selectVideo(state.videos[0]?.id ?? null);
    } else if (state.selectedId) {
      const previousIndex = state.video?.index?.dir;
      const video = await api.video(state.selectedId);
      if (video.id === state.selectedId) {
        state.video = video;
        if (video.index?.dir !== previousIndex) await loadTranscript();
        if (state.searchId && ACTIVE.includes(state.search?.status)) {
          const search = await api.search(state.selectedId, state.searchId);
          const finished = search.status === 'done';
          state.search = search;
          if (finished) {
            render();
            seekToFirstMatch();
          }
        }
      }
    }
    render();
  } catch {
    if (!state.offline) {
      state.offline = true;
      renderSystem();
      renderBanner();
    }
  }
  schedulePoll();
}

// ---------------------------------------------------------------- render

function render() {
  renderLibrary();
  renderSystem();
  renderBanner();
  els.empty.hidden = Boolean(state.video);
  els.workspace.hidden = !state.video;
  if (!state.video) return;
  renderHeader();
  renderPlayer();
  renderSearchForm();
  renderResults();
  renderTimeline();
  renderTabs();
  tickElapsed();
}

function status(kind, text) {
  return h('span', { class: `status status-${kind}` }, h('span', { class: 'status-dot' }), text);
}

function miniProgress(fraction) {
  return h('span', { class: 'mini-progress', 'aria-hidden': 'true' }, h('span', { style: { width: pct(fraction) } }));
}

function jobProgress(job) {
  if (!job?.stage) return 0;
  const count = Math.max(1, job.stages.length);
  const index = Math.max(0, job.stages.indexOf(job.stage));
  const task = [...job.tasks].reverse().find((t) => t.total);
  return Math.min(1, (index + (task ? Math.min(1, task.done / task.total) : 0)) / count);
}

function videoStatus(video) {
  const indexJob = video.jobs.find((j) => j.kind === 'index');
  const searchJob = video.jobs.find((j) => j.kind === 'search');
  if (indexJob?.status === 'running') return status('busy', `${video.index ? 'Re-indexing' : 'Indexing'} · ${pct(jobProgress(indexJob))}`);
  if (indexJob) return status('muted', 'Queued for indexing');
  if (searchJob) return status('busy', searchJob.status === 'running' ? 'Searching' : 'Search queued');
  if (video.status === 'ready') return status(video.error ? 'warning' : 'good', video.error ? 'Ready · re-index failed' : 'Ready');
  if (video.status === 'failed') return status('critical', 'Indexing failed');
  if (['cancelled', 'interrupted'].includes(video.status)) return status('warning', 'Indexing paused');
  return status('muted', 'Not indexed');
}

function renderLibrary() {
  const key = [
    state.selectedId,
    state.uploads.map((u) => [u.key, Math.round(u.progress * 100)]),
    state.videos.map((v) => [v.id, v.name, v.status, v.error, v.jobs.map((j) => [j.kind, j.status, j.stage, j.tasks])]),
  ];
  if (!changed('library', key)) return;
  els.libraryCount.textContent = state.videos.length ? String(state.videos.length) : '';
  const items = [
    ...state.uploads.map((upload) => h('li', {},
      h('div', { class: 'library-item is-uploading' },
        h('span', { class: 'thumb thumb-upload' }, icon('upload', 18)),
        h('span', { class: 'library-text' },
          h('span', { class: 'library-name', title: upload.name }, upload.name),
          status('busy', `Uploading · ${pct(upload.progress)}`)),
        miniProgress(upload.progress)))),
    ...state.videos.map((video) => {
      const selected = video.id === state.selectedId;
      const indexJob = video.jobs.find((j) => j.kind === 'index' && j.status === 'running');
      return h('li', {}, h('button', {
        type: 'button',
        class: `library-item${selected ? ' is-selected' : ''}`,
        'aria-current': selected ? 'true' : null,
        onclick: () => { if (video.id !== state.selectedId) selectVideo(video.id); },
      },
      h('span', { class: 'thumb' },
        h('img', { src: video.thumbnail_url, alt: '', loading: 'lazy', onerror: (event) => event.target.remove() }),
        h('span', { class: 'thumb-duration' }, clockTime(video.duration))),
      h('span', { class: 'library-text' },
        h('span', { class: 'library-name', title: video.name }, video.name),
        videoStatus(video)),
      indexJob ? miniProgress(jobProgress(indexJob)) : null));
    }),
  ];
  if (!items.length) items.push(h('li', { class: 'library-empty' }, 'Videos you add appear here.'));
  els.library.replaceChildren(...items);
}

function renderSystem() {
  const sys = state.system;
  let kind = 'muted';
  let title = 'Checking local models';
  let detail = '';
  if (state.offline) {
    [kind, title, detail] = ['critical', 'App server unreachable', 'Is start.ps1 still running?'];
  } else if (sys && !sys.ollama.running) {
    [kind, title, detail] = ['critical', 'Model server stopped', 'Run .\\start.ps1'];
  } else if (sys?.missing_models.length) {
    [kind, title, detail] = ['warning', 'Model not installed', sys.missing_models.join(', ')];
  } else if (sys && !sys.ffmpeg) {
    [kind, title, detail] = ['warning', 'FFmpeg not found', 'Add FFmpeg to PATH'];
  } else if (sys) {
    [kind, title, detail] = ['good', 'Local models ready', `${sys.settings.vision} · ${sys.devices?.gpu ?? 'CPU'}`];
  }
  if (!changed('system', [kind, title, detail])) return;
  els.systemPill.replaceChildren(
    h('span', { class: `status status-${kind}` }, h('span', { class: 'status-dot' })),
    h('span', { class: 'pill-text' }, h('strong', {}, title), detail ? h('small', { title: detail }, detail) : null),
    icon('settings', 16));
}

function renderBanner() {
  const sys = state.system;
  let kind = 'warning';
  let parts = null;
  if (state.offline) {
    kind = 'critical';
    parts = ['Lost connection to the Moments server. Restart it with ', h('code', {}, '.\\start.ps1'), '; this page reconnects on its own.'];
  } else if (sys && !sys.ollama.running) {
    kind = 'critical';
    parts = ['The local model server isn’t running, so indexing and search are paused. Start it with ', h('code', {}, '.\\start.ps1'), '.'];
  } else if (sys?.missing_models.length) {
    parts = ['A selected model isn’t installed. Run ', h('code', {}, `ollama pull ${sys.missing_models[0]}`), ' or choose another model in settings.'];
  }
  const key = [kind, parts?.map((p) => (typeof p === 'string' ? p : p.textContent))];
  if (!changed('banner', key)) return;
  els.banner.hidden = !parts;
  els.banner.className = `banner banner-${kind}`;
  if (parts) {
    els.banner.replaceChildren(icon('alert', 18), h('span', {}, ...parts),
      state.offline ? null : h('button', { type: 'button', class: 'button button-ghost button-small', onclick: openSettings }, 'Settings'));
  }
}

function renderHeader() {
  const v = state.video;
  const indexJob = v.jobs.find((j) => j.kind === 'index');
  const key = [v.id, v.name, v.status, v.error, v.index?.dir, v.index_outdated, v.jobs.map((j) => [j.kind, j.status, j.stage, j.tasks])];
  if (!changed('header', key)) return;
  els.title.textContent = v.name;
  els.title.title = v.name;
  const stats = v.index?.stats ?? {};
  els.meta.replaceChildren(
    h('span', { class: 'chip chip-mono' }, icon('clock', 13), clockTime(v.duration)),
    h('span', { class: 'chip' }, `${v.width}×${v.height}`),
    h('span', { class: 'chip' }, formatBytes(v.size)),
    v.index ? h('span', { class: 'chip' }, plural(stats.scene_records ?? 0, 'scene description')) : null,
    v.index ? h('span', { class: 'chip' }, v.index.has_transcript ? plural(stats.speech_segments ?? 0, 'speech segment') : 'No speech') : null,
    v.index?.models?.scene_model ? h('span', { class: 'chip', title: 'Scene model used for indexing' }, icon('cpu', 13), v.index.models.scene_model) : null,
    videoStatus(v));
  const actions = [];
  if (!indexJob && v.index && (v.index_outdated || v.error)) {
    actions.push(h('button', {
      type: 'button', class: 'button button-ghost button-small', onclick: reindex,
      title: v.index_outdated ? 'Model settings changed since this video was indexed' : v.error,
    }, icon('refresh', 14), v.index_outdated ? 'Re-index with new settings' : 'Retry re-index'));
  }
  actions.push(h('button', { type: 'button', class: 'icon-button', title: 'Delete video', 'aria-label': 'Delete video', onclick: deleteVideo }, icon('trash', 17)));
  els.actions.replaceChildren(...actions);
}

function renderPlayer() {
  const v = state.video;
  const source = `${v.media_url}?preview=${v.has_preview ? 1 : 0}`;
  if (els.video.dataset.src !== source) {
    els.video.dataset.src = source;
    els.video.poster = v.thumbnail_url;
    els.video.src = source;
  }
  const job = v.jobs.find((j) => j.kind === 'index');
  const dismissed = state.overlayDismissed.has(v.id);
  const key = [v.id, v.status, v.error, Boolean(v.index), dismissed, job && [job.status, job.stage, job.stages, job.tasks, job.model, job.cancel_requested]];
  if (!changed('overlay', key)) return;
  let card = null;
  if (!v.index && job && !dismissed) card = indexingCard(v, job);
  else if (!v.index && !job) card = indexIssueCard(v);
  els.overlay.hidden = !card;
  if (card) els.overlay.replaceChildren(card);
}

function indexingCard(v, job) {
  const queued = job.status === 'queued';
  return h('div', { class: 'overlay-card' },
    h('div', { class: 'progress-head' },
      h('span', { class: 'spinner' }),
      h('div', { class: 'progress-text' },
        h('h2', {}, queued ? 'Waiting to index' : 'Indexing this video'),
        h('p', {}, queued
          ? 'Another job is using the local models. Indexing starts automatically.'
          : 'Frames, speech, and scenes are read once, so every search afterwards is fast.')),
      h('span', { class: 'elapsed', dataset: { since: job.started_at ?? '' } })),
    stepList(job),
    h('div', { class: 'overlay-actions' },
      h('button', { type: 'button', class: 'button button-ghost button-small', onclick: () => { state.overlayDismissed.add(v.id); renderPlayer(); } }, 'Watch while it indexes'),
      h('button', { type: 'button', class: 'button button-ghost button-small', disabled: job.cancel_requested, onclick: () => cancelJob(job) },
        job.cancel_requested ? 'Stopping…' : 'Stop')));
}

function indexIssueCard(v) {
  const failed = v.status === 'failed';
  const title = failed ? 'Indexing failed' : v.status === 'uploaded' ? 'This video isn’t indexed yet' : 'Indexing paused';
  const message = v.error || (failed ? 'Something went wrong while indexing.' : 'Resume to continue; finished scene descriptions are kept.');
  return h('div', { class: 'overlay-card' },
    h('h2', { class: `issue-title issue-${failed ? 'critical' : 'warning'}` }, icon(failed ? 'alert' : 'clock', 18), title),
    h('p', {}, message),
    h('div', { class: 'overlay-actions' },
      h('button', { type: 'button', class: 'button button-primary button-small', onclick: reindex }, icon('refresh', 14), failed ? 'Try again' : 'Resume indexing')));
}

function stepList(job) {
  const current = job.stages.indexOf(job.stage);
  return h('ol', { class: 'steps' }, job.stages.map((name, index) => {
    const phase = job.status === 'done' || (current >= 0 && index < current) ? 'done' : index === current ? 'active' : 'pending';
    const task = phase === 'active' ? [...job.tasks].reverse().find((t) => t.total) : null;
    return h('li', { class: `step step-${phase}` },
      h('span', { class: 'step-dot' }, phase === 'done' ? icon('check', 12) : null),
      h('div', { class: 'step-body' },
        h('span', { class: 'step-name' }, name),
        task ? h('div', { class: 'step-task' },
          h('div', { class: 'bar' }, h('span', { style: { width: pct(Math.min(1, task.done / task.total)) } })),
          h('span', { class: 'step-meta' }, `${task.name} · ${Math.min(task.done, task.total)} of ${task.total}`)) : null,
        phase === 'active' && job.model ? h('span', { class: 'step-meta' }, `Model: ${job.model}`) : null));
  }));
}

function renderSearchForm() {
  const ready = Boolean(state.video.index);
  const running = ACTIVE.includes(state.search?.status);
  els.form.classList.toggle('is-disabled', !ready);
  els.input.disabled = !ready;
  els.submit.disabled = !ready || running;
  els.submit.textContent = running ? 'Searching…' : 'Search';
  els.input.placeholder = ready
    ? 'Describe a moment, a spoken phrase, or text on screen…'
    : 'Search unlocks when indexing finishes';
}

function setMode(mode) {
  state.mode = mode;
  storage.set('moments.mode', mode);
  for (const button of els.modeButtons) button.setAttribute('aria-checked', String(button.dataset.mode === mode));
  els.optRefine.disabled = mode === 'quick';
  if (mode === 'quick' && Number(els.optCandidates.value) === 20) setCandidates(12);
  if (mode === 'verified' && Number(els.optCandidates.value) === 12) setCandidates(20);
}

function setCandidates(value) {
  els.optCandidates.value = String(value);
  els.optCandidatesOut.textContent = String(value);
}

function renderResults() {
  const v = state.video;
  const s = state.search;
  const job = s?.job;
  const key = [v.id, Boolean(v.index), v.jobs.length, s?.id, s?.status, s?.error,
    job && ACTIVE.includes(s.status) ? [job.status, job.stage, job.stages, job.tasks, job.model, job.cancel_requested] : null];
  if (!changed('results', key)) return;
  let content;
  if (!v.index) content = [lockedCard(v)];
  else if (!s) content = [introCard()];
  else if (ACTIVE.includes(s.status)) content = [progressCard(s)];
  else if (s.status === 'failed') content = [issueCard(s, 'Search failed', s.error || 'Unknown error.')];
  else if (s.status === 'cancelled') content = [issueCard(s, 'Search cancelled', 'The search stopped before it finished.')];
  else content = resultsContent(s);
  els.results.replaceChildren(...content);
  els.results.scrollTop = 0;
}

function lockedCard(v) {
  const indexing = v.jobs.some((j) => j.kind === 'index');
  return h('div', { class: 'card intro' },
    h('h2', {}, indexing ? 'Search unlocks after indexing' : 'Index this video to search it'),
    h('p', {}, indexing
      ? 'Indexing runs once per video. You can watch the video while it works.'
      : 'Moments needs to read the frames, speech, and scenes before it can answer questions.'));
}

function introCard() {
  return h('div', { class: 'card intro' },
    h('h2', {}, 'Ask about this video'),
    h('p', {}, 'Describe what happens, what someone says, or text that appears on screen. Answers come back as timestamped clips.'),
    h('div', { class: 'examples' }, EXAMPLES.map((text) => h('button', {
      type: 'button', class: 'example', onclick: () => { els.input.value = text; els.input.focus(); },
    }, text))),
    h('div', { class: 'mode-cards' },
      h('div', { class: 'mode-card' }, icon('shield', 18), h('div', {}, h('strong', {}, 'Verified'),
        'Checks each candidate with the local vision model and tightens the clip boundaries. Usually a minute or more.')),
      h('div', { class: 'mode-card' }, icon('bolt', 18), h('div', {}, h('strong', {}, 'Quick'),
        'Ranks likely moments straight from the index in seconds. Results aren’t checked.'))));
}

function progressCard(s) {
  const job = s.job;
  return h('div', { class: 'card' },
    h('div', { class: 'progress-head' },
      h('span', { class: 'spinner' }),
      h('div', { class: 'progress-text' },
        h('div', { class: 'progress-title' }, s.status === 'queued' ? 'Waiting for the models' : 'Searching'),
        h('div', { class: 'progress-query', title: s.query }, `“${s.query}”`)),
      h('span', { class: 'elapsed', dataset: { since: job?.started_at ?? '' } }),
      job ? h('button', { type: 'button', class: 'button button-ghost button-small', disabled: job.cancel_requested, onclick: () => cancelJob(job) },
        job.cancel_requested ? 'Stopping…' : 'Cancel') : null),
    job ? stepList(job) : null,
    h('p', { class: 'progress-note' }, s.mode === 'verified'
      ? 'Verified search checks candidates with the local vision model. Expect one to a few minutes.'
      : 'Quick search ranks moments from the index without vision checks.'));
}

function issueCard(s, title, message) {
  return h('div', { class: 'card issue-card' },
    h('h2', { class: 'issue-title issue-critical' }, icon('alert', 18), title),
    h('p', {}, message),
    h('button', { type: 'button', class: 'button button-ghost button-small', onclick: () => runSearch(s.query, s.mode) }, icon('refresh', 14), 'Run again'));
}

function modeBadge(mode) {
  return h('span', { class: 'mode-badge' }, icon(mode === 'quick' ? 'bolt' : 'shield', 13), mode === 'quick' ? 'Quick' : 'Verified');
}

function resultsContent(s) {
  const result = s.result || {};
  const matches = result.matches || [];
  const summary = h('div', { class: 'card results-summary' },
    h('div', { class: 'results-head' },
      h('div', {},
        h('h2', {}, plural(matches.length, 'moment')),
        h('div', { class: 'results-sub' }, modeBadge(s.mode), h('span', {}, '·'), h('span', {}, formatElapsed(s.elapsed)), h('span', {}, '·'), h('span', {}, relativeTime(s.created_at)))),
      h('a', {
        class: 'icon-button', href: `/api/videos/${state.video.id}/searches/${s.id}`, download: `moments-${s.id}.json`,
        title: 'Export results as JSON', 'aria-label': 'Export results as JSON',
      }, icon('download', 17))),
    h('p', { class: 'results-query' }, `“${s.query}”`),
    result.text_entities?.length ? h('div', { class: 'text-entities' },
      h('div', { class: 'section-title' }, 'Text found'),
      result.text_entities.map((entity) => h('div', { class: 'entity' },
        h('span', { class: 'entity-text' }, entity.text),
        h('span', { class: 'entity-meta' }, `${plural(entity.appearances.length, 'appearance')} · ${pct(entity.confidence)}`)))) : null);
  if (!matches.length) {
    return [summary, h('div', { class: 'card placeholder' }, icon('search', 22),
      h('strong', {}, s.mode === 'verified' ? 'Nothing passed verification' : 'No likely moments found'),
      h('p', {}, s.mode === 'verified'
        ? 'Try describing it differently, lowering the minimum confidence, or running a Quick search to see near misses.'
        : 'Try other wording, or describe what is visible or said.'))];
  }
  return [summary, ...matches.map((match, index) => matchCard(match, index, s))];
}

function matchCard(match, index, s) {
  const confidence = Number(match.confidence ?? 0);
  const frames = match.frames || [];
  const thumbnail = match.best_frame_url || frames[Math.floor((frames.length - 1) / 2)]?.url;
  const description = match.description || 'Matching moment';
  const facts = [
    ['Who', match.actor_description],
    ['Seen', match.visual_evidence],
    ['Verifier', match.pro_reason],
    ['Where', match.target_description],
    ['Model', match.verification_model],
    ['Full note', description.length > 150 ? description : null],
  ].filter(([, value]) => value);
  const hasDetails = facts.length || frames.length || match.region_crop_url || match.clip_url;
  return h('article', { class: `match${index === state.activeMatch ? ' is-active' : ''}`, dataset: { index } },
    h('button', { type: 'button', class: 'match-thumb', onclick: () => playMatch(index), 'aria-label': `Play moment ${index + 1}` },
      thumbnail ? h('img', { src: thumbnail, alt: '', loading: 'lazy' }) : null,
      h('span', { class: 'match-play' }, icon('play', 14)),
      h('span', { class: 'match-length' }, `${Math.max(0, match.end - match.start).toFixed(1)}s`)),
    h('div', { class: 'match-body' },
      h('div', { class: 'match-top' },
        h('span', { class: 'match-number' }, index + 1),
        h('button', { type: 'button', class: 'match-time', onclick: () => playMatch(index) }, `${preciseTime(match.start)} – ${preciseTime(match.end)}`)),
      match.extracted_text ? h('div', { class: 'match-text' }, match.extracted_text) : null,
      h('p', { class: 'match-desc' }, description),
      h('div', { class: 'meter' },
        h('span', { class: 'meter-track', 'aria-hidden': 'true' }, h('span', { class: 'meter-fill', style: { width: pct(Math.min(1, confidence)) } })),
        h('span', {}, s.mode === 'quick' ? `Evidence ${pct(confidence)} · unverified` : `${pct(confidence)} confidence`)),
      hasDetails ? h('details', { class: 'match-details' },
        h('summary', {}, 'Evidence and clip', icon('chevron', 14)),
        h('div', { class: 'details-body' },
          facts.length ? h('dl', { class: 'facts' }, facts.flatMap(([label, value]) => [h('dt', {}, label), h('dd', {}, value)])) : null,
          match.region_crop_url ? h('img', { class: 'crop', src: match.region_crop_url, alt: 'Region where the text was read' }) : null,
          frames.length ? h('div', { class: 'filmstrip' }, frames.slice(0, 8).map((frame) => h('button', {
            type: 'button', class: 'frame', title: `Jump to ${preciseTime(frame.timestamp)}`, onclick: () => seek(frame.timestamp),
          }, h('img', { src: frame.url, alt: `Frame at ${preciseTime(frame.timestamp)}`, loading: 'lazy' })))) : null,
          match.clip_url ? h('a', { class: 'button button-ghost button-small', href: match.clip_url, download: `moment-${index + 1}.mp4` },
            icon('download', 14), 'Download clip') : null)) : null));
}

function renderTimeline() {
  const v = state.video;
  const result = state.search?.status === 'done' ? state.search.result : null;
  if (!changed('timeline', [v.id, state.search?.id, state.search?.status, state.transcript.length])) return;
  const diagnostics = result?.diagnostics ?? {};
  timeline.set({
    duration: v.duration,
    evidence: diagnostics.evidence_map ?? [],
    candidates: diagnostics.candidates ?? [],
    matches: result?.matches ?? [],
    speech: state.transcript,
  });
  timeline.select(state.activeMatch);
  timeline.setTime(els.video.currentTime || 0);
}

function renderTabs() {
  const counts = { transcript: state.transcript.length || '', history: state.video.searches?.length || '' };
  for (const tab of els.tabs) {
    const active = tab.dataset.tab === state.tab;
    tab.classList.toggle('is-active', active);
    tab.setAttribute('aria-selected', String(active));
    tab.tabIndex = active ? 0 : -1;
    const count = tab.querySelector('.tab-count');
    if (count) count.textContent = String(counts[tab.dataset.tab] ?? '');
  }
  if (state.tab === 'plan' && changed('tab', ['plan', state.video.id, state.search?.id, state.search?.status])) renderPlan();
  if (state.tab === 'transcript' && changed('tab', ['transcript', state.video.id, Boolean(state.video.index), state.transcript.length])) renderTranscript();
  if (state.tab === 'history' && changed('tab', ['history', state.video.id, state.video.searches, state.searchId])) renderHistory();
}

function placeholder(iconName, title, text) {
  return h('div', { class: 'placeholder' }, icon(iconName, 22), h('strong', {}, title), h('p', {}, text));
}

function section(title, ...children) {
  return h('section', { class: 'plan-section' }, h('h3', {}, title), ...children);
}

function stat(value, label) {
  return h('div', { class: 'stat' }, h('div', { class: 'stat-label' }, label), h('div', { class: 'stat-value', title: String(value) }, value));
}

function renderPlan() {
  const search = state.search;
  const result = search?.status === 'done' ? search.result : null;
  if (!result) {
    els.tabPanel.replaceChildren(placeholder('sparkle', 'How your question is searched',
      search && ACTIVE.includes(search.status)
        ? 'The plan appears here when the search finishes.'
        : 'Run a search to see how it is split into visual, scene, and speech evidence, and how candidates were narrowed down.'));
    return;
  }
  const plan = result.plan || {};
  const d = result.diagnostics || {};
  const textRoute = plan.executor === 'visual_text_extraction';
  const models = result.model_backend || {};
  const blocks = [h('div', { class: 'plan-summary' },
    stat(textRoute ? 'Read on-screen text' : 'Find events', 'Route'),
    stat(plan.return_mode === 'all' ? 'Every occurrence' : 'Single best match', 'Returns'),
    stat(formatElapsed(search.elapsed) || '—', 'Search time'),
    stat(models.planner ?? '—', 'Planner'))];
  if (plan.routing_reason) blocks.push(section('Why this route', h('p', {}, plan.routing_reason)));

  if (textRoute) {
    const rows = [['Object', plan.target_object], ['Region', plan.target_region], ['Text', plan.text_description], ['Instruction', plan.extraction_instruction]].filter(([, value]) => value);
    blocks.push(section('Reading target', h('dl', { class: 'facts facts-wide' }, rows.flatMap(([label, value]) => [h('dt', {}, label), h('dd', {}, value)]))));
    if (d.num_coarse_candidates != null) {
      blocks.push(section('Search funnel', h('div', { class: 'funnel' },
        funnelStep(d.num_coarse_candidates, 'Text sightings'), funnelStep(result.num_matches, 'Returned'))));
    }
  } else {
    if (plan.target_event) blocks.push(section('Target event', h('p', {}, plan.target_event)));
    blocks.push(section('Channel weights', h('div', { class: 'bars' }, Object.entries(CHANNEL_LABELS).map(([key, label]) => {
      const weight = Number(plan.weights?.[key] ?? 0);
      return h('div', { class: 'bar-row' },
        h('span', { class: 'bar-label' }, label),
        h('span', { class: 'bar-track', 'aria-hidden': 'true' }, h('span', { class: 'bar-fill', style: { width: pct(weight) } })),
        h('span', { class: 'bar-value' }, pct(weight)));
    }))));
    const predicates = plan.evidence_predicates || [];
    if (predicates.length) {
      blocks.push(section('Evidence it looked for', h('ul', { class: 'predicates' }, predicates.map((p) => h('li', {},
        h('div', { class: 'predicate-head' }, h('span', { class: 'role' }, p.role), p.required ? h('span', { class: 'required' }, 'Required') : null),
        h('p', {}, p.description),
        h('div', { class: 'chips' }, (p.modalities || []).map((m) => h('span', { class: 'chip chip-small' }, CHANNEL_LABELS[m] ?? m))))))));
    }
    const definition = plan.event_definition || {};
    if (definition.counts_as_match?.length || definition.does_not_count?.length) {
      blocks.push(section('What counts', h('div', { class: 'definition' },
        h('div', {}, h('h4', {}, icon('check', 14), 'Counts'), h('ul', {}, (definition.counts_as_match || []).map((x) => h('li', {}, x)))),
        h('div', {}, h('h4', {}, icon('x', 14), 'Doesn’t count'), h('ul', {}, (definition.does_not_count || []).map((x) => h('li', {}, x)))))));
    }
    const negatives = plan.negative_evidence || [];
    if (negatives.length) {
      blocks.push(section('Confounders it rejects', h('ul', { class: 'plain-list' }, negatives.map((text) => h('li', {}, text)))));
    }
    const ordering = plan.ordering || [];
    if (ordering.length) {
      blocks.push(section('Required order', h('ul', { class: 'plain-list' }, ordering.map((rule) =>
        h('li', {}, `${rule.first} before ${rule.then}${rule.description ? ` - ${rule.description}` : ''}`)))));
    }
    const steps = [
      [d.num_candidates, 'Candidate regions'],
      [d.num_flash_instances, 'First vision pass'],
      [d.num_pro_instances, 'Confirmed'],
      [d.num_final_matches, 'Returned'],
    ].filter(([n]) => n != null);
    if (steps.length) blocks.push(section('Search funnel', h('div', { class: 'funnel' }, steps.map(([n, label]) => funnelStep(n, label)))));
  }
  els.tabPanel.replaceChildren(h('div', { class: 'plan' }, blocks));
}

function funnelStep(count, label) {
  return h('div', { class: 'funnel-step' }, h('span', { class: 'funnel-n' }, count), h('span', { class: 'funnel-label' }, label));
}

function renderTranscript() {
  if (!state.video.index) {
    els.tabPanel.replaceChildren(placeholder('clock', 'Transcript on the way', 'Speech is transcribed while the video indexes.'));
    return;
  }
  if (!state.transcript.length) {
    els.tabPanel.replaceChildren(placeholder('text', 'No speech detected', 'Searches for this video rely on visual frames and scene descriptions.'));
    return;
  }
  const list = h('ol', { class: 'transcript' });
  const fill = () => {
    const query = state.transcriptFilter.trim().toLowerCase();
    const rows = state.transcript
      .map((segment, index) => [segment, index])
      .filter(([segment]) => !query || segment.text.toLowerCase().includes(query))
      .map(([segment, index]) => h('li', { dataset: { index } },
        h('button', { type: 'button', class: 'transcript-row', onclick: () => seek(segment.start) },
          h('span', { class: 'transcript-time' }, clockTime(segment.start)),
          h('span', { class: 'transcript-text' }, highlight(segment.text, query)))));
    list.replaceChildren(...(rows.length ? rows : [h('li', { class: 'transcript-empty' }, 'No lines match.')]));
    lastTranscriptIndex = -1;
    highlightTranscript(els.video.currentTime);
  };
  const filter = h('input', {
    class: 'input', type: 'search', placeholder: 'Filter transcript', 'aria-label': 'Filter transcript', value: state.transcriptFilter,
    oninput: (event) => { state.transcriptFilter = event.target.value; fill(); },
  });
  els.tabPanel.replaceChildren(h('div', { class: 'transcript-panel' }, filter, list));
  fill();
}

function highlightTranscript(time) {
  if (state.tab !== 'transcript' || !state.transcript.length) return;
  let low = 0;
  let high = state.transcript.length - 1;
  let index = -1;
  while (low <= high) {
    const mid = (low + high) >> 1;
    if (state.transcript[mid].start <= time) { index = mid; low = mid + 1; } else high = mid - 1;
  }
  if (index === lastTranscriptIndex) return;
  lastTranscriptIndex = index;
  els.tabPanel.querySelector('.transcript li.is-current')?.classList.remove('is-current');
  const item = els.tabPanel.querySelector(`.transcript li[data-index="${index}"]`);
  if (!item) return;
  item.classList.add('is-current');
  const panel = els.tabPanel;
  if (!els.video.paused && (item.offsetTop < panel.scrollTop || item.offsetTop > panel.scrollTop + panel.clientHeight - 48)) {
    panel.scrollTop = item.offsetTop - 64;
  }
}

function renderHistory() {
  const searches = state.video.searches || [];
  if (!searches.length) {
    els.tabPanel.replaceChildren(placeholder('clock', 'No searches yet', 'Every search on this video is saved here with its results.'));
    return;
  }
  els.tabPanel.replaceChildren(h('ul', { class: 'history' }, searches.map((row) => h('li', { class: row.id === state.searchId ? 'is-selected' : '' },
    h('button', { type: 'button', class: 'history-row', onclick: () => openSearch(row.id) },
      h('span', { class: 'history-query', title: row.query }, row.query),
      h('span', { class: 'history-meta' },
        modeBadge(row.mode),
        row.status === 'done'
          ? h('span', {}, plural(row.num_matches ?? 0, 'moment'))
          : status(ACTIVE.includes(row.status) ? 'busy' : row.status === 'failed' ? 'critical' : 'warning', row.status === 'running' ? 'Running' : row.status[0].toUpperCase() + row.status.slice(1)),
        row.elapsed ? h('span', {}, formatElapsed(row.elapsed)) : null,
        h('span', {}, relativeTime(row.created_at)))),
    h('button', { type: 'button', class: 'icon-button', title: 'Delete search', 'aria-label': `Delete search “${row.query}”`, onclick: () => deleteSearch(row.id) }, icon('trash', 15))))));
}

// --------------------------------------------------------------- actions

async function runSearch(query, mode = state.mode) {
  const v = state.video;
  if (!v?.index) return;
  els.submit.disabled = true;
  try {
    const search = await api.createSearch(v.id, {
      query,
      mode,
      options: {
        refine_boundaries: els.optRefine.checked,
        min_confidence: Number(els.optConfidence.value),
        max_candidates: Number(els.optCandidates.value),
      },
    });
    state.search = search;
    state.searchId = search.id;
    state.activeMatch = -1;
    state.video.searches = [{ id: search.id, query: search.query, mode: search.mode, status: search.status, created_at: search.created_at }, ...(state.video.searches || [])];
    els.options.open = false;
    updateHash();
    render();
    schedulePoll(600);
  } catch (error) {
    toast(error.message, 'error');
  } finally {
    renderSearchForm();
  }
}

async function openSearch(searchId) {
  await loadSearch(searchId);
  state.activeMatch = -1;
  updateHash();
  render();
  seekToFirstMatch();
}

async function deleteSearch(searchId) {
  try {
    await api.deleteSearch(state.video.id, searchId);
    state.video.searches = state.video.searches.filter((row) => row.id !== searchId);
    if (state.searchId === searchId) {
      state.search = null;
      state.searchId = null;
      updateHash();
    }
    render();
  } catch (error) {
    toast(error.message, 'error');
  }
}

async function cancelJob(job) {
  try {
    await api.cancelJob(job.id);
    job.cancel_requested = true;
    rendered.results = null;
    rendered.overlay = null;
    render();
    schedulePoll(400);
  } catch (error) {
    toast(error.message, 'error');
  }
}

async function reindex() {
  try {
    await api.reindex(state.video.id);
    state.overlayDismissed.delete(state.video.id);
    toast('Indexing started.', 'info');
    schedulePoll(300);
  } catch (error) {
    toast(error.message, 'error');
  }
}

async function deleteVideo() {
  const v = state.video;
  const searches = v.searches?.length || 0;
  const confirmed = await confirmDialog({
    title: 'Delete this video?',
    message: `“${v.name}”, its index, and ${searches} saved ${searches === 1 ? 'search' : 'searches'} will be removed from this computer.`,
    confirmLabel: 'Delete video',
  });
  if (!confirmed) return;
  // Release the media file first; Windows refuses to delete files that are open.
  els.video.pause();
  els.video.removeAttribute('src');
  els.video.load();
  delete els.video.dataset.src;
  try {
    await api.deleteVideo(v.id);
    toast('Video deleted.', 'success');
    await loadVideos();
    await selectVideo(state.videos[0]?.id ?? null);
  } catch (error) {
    toast(error.message, 'error');
    render();
  }
}

function addFiles(files) {
  for (const file of files) {
    const upload = { key: crypto.randomUUID?.() ?? `${Date.now()}-${Math.random()}`, name: file.name, progress: 0 };
    state.uploads.push(upload);
    renderLibrary();
    uploadVideo(file, (progress) => { upload.progress = progress; renderLibrary(); }).promise
      .then(async (video) => {
        state.uploads = state.uploads.filter((u) => u !== upload);
        await loadVideos();
        await selectVideo(video.id);
        toast(video.status === 'ready' ? `${video.name} is already in your library.` : `Added ${video.name}. Indexing has started.`, 'success');
        schedulePoll(500);
      })
      .catch((error) => {
        state.uploads = state.uploads.filter((u) => u !== upload);
        toast(`${file.name}: ${error.message}`, 'error');
        renderLibrary();
      });
  }
  schedulePoll(1000);
}

// -------------------------------------------------------------- playback

function seek(time, keepSegment = false) {
  if (!Number.isFinite(time)) return;
  if (!keepSegment) state.segmentEnd = null;
  programmaticSeek = true;
  els.video.currentTime = Math.max(0, time);
  timeline.setTime(time);
}

function playMatch(index) {
  const match = state.search?.result?.matches?.[index];
  if (!match) return;
  state.activeMatch = index;
  state.segmentEnd = match.end;
  seek(match.start, true);
  els.video.play().catch(() => { /* autoplay can be blocked until the user interacts */ });
  for (const card of els.results.querySelectorAll('.match')) card.classList.toggle('is-active', Number(card.dataset.index) === index);
  els.results.querySelector(`.match[data-index="${index}"]`)?.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
  timeline.select(index);
}

function seekToFirstMatch() {
  const first = state.search?.status === 'done' ? state.search.result?.matches?.[0] : null;
  if (first && els.video.paused) seek(first.start);
}

function onTimeUpdate() {
  const time = els.video.currentTime;
  timeline.setTime(time);
  if (state.segmentEnd != null && time >= state.segmentEnd) {
    els.video.pause();
    state.segmentEnd = null;
  }
  highlightTranscript(time);
}

function frameLoop() {
  onTimeUpdate();
  if (!els.video.paused && !els.video.ended) requestAnimationFrame(frameLoop);
}

// --------------------------------------------------------------- dialogs

function toast(message, kind = 'info') {
  const el = h('div', { class: `toast toast-${kind}`, role: kind === 'error' ? 'alert' : 'status' },
    icon(kind === 'error' ? 'alert' : kind === 'success' ? 'check' : 'info', 16), h('span', {}, message));
  els.toasts.append(el);
  setTimeout(() => el.remove(), kind === 'error' ? 8000 : 4500);
}

function confirmDialog({ title, message, confirmLabel }) {
  return new Promise((resolve) => {
    const dialog = els.confirm;
    const finish = (value) => { if (dialog.open) dialog.close(); resolve(value); };
    dialog.replaceChildren(h('div', { class: 'dialog-body' },
      h('h2', {}, title),
      h('p', { class: 'dialog-text' }, message),
      h('div', { class: 'dialog-actions' },
        h('button', { type: 'button', class: 'button button-ghost', onclick: () => finish(false) }, 'Cancel'),
        h('button', { type: 'button', class: 'button button-danger', onclick: () => finish(true) }, confirmLabel))));
    dialog.addEventListener('cancel', () => resolve(false), { once: true });
    dialog.showModal();
  });
}

function openSettings() {
  const sys = state.system;
  const current = sys?.settings ?? {};
  const models = sys?.ollama?.models ?? [];
  const devices = sys?.devices ?? {};
  const options = (id, capability) => h('datalist', { id },
    models.filter((m) => !capability || m.capabilities.includes(capability))
      .map((m) => h('option', { value: m.name }, [m.parameter_size, formatBytes(m.size)].filter(Boolean).join(' · '))));
  const field = (name, label, help, list, wide = false) => h('label', { class: `field${wide ? ' field-wide' : ''}` },
    h('span', { class: 'field-label' }, label),
    h('input', { class: 'input', name, value: current[name] ?? '', list, required: true, spellcheck: 'false', autocomplete: 'off' }),
    h('small', { class: 'field-help' }, help));
  const framesOut = h('output', {}, String(current.frame_limit ?? 12));
  const frames = h('input', {
    type: 'range', name: 'frame_limit', min: '4', max: '24', step: '1', value: String(current.frame_limit ?? 12),
    oninput: (event) => { framesOut.textContent = event.target.value; },
  });
  const statusItem = (label, ok, detail) => h('div', { class: 'status-item' },
    h('strong', {}, h('span', { class: `status status-${ok ? 'good' : 'critical'}` }, h('span', { class: 'status-dot' })), label), detail);

  const form = h('form', {
    class: 'dialog-body',
    onsubmit: async (event) => {
      event.preventDefault();
      const data = Object.fromEntries(new FormData(form));
      data.frame_limit = Number(data.frame_limit);
      try {
        const saved = await api.saveSettings(data);
        const reindexNeeded = saved.vision !== current.vision || saved.frame_limit !== current.frame_limit;
        toast(reindexNeeded ? 'Settings saved. Re-index videos to apply the new scene settings.' : 'Settings saved.', 'success');
        els.settings.close();
        await loadSystem();
        poll();
      } catch (error) {
        toast(error.message, 'error');
      }
    },
  },
  h('div', { class: 'dialog-head' },
    h('h2', {}, 'Local models'),
    h('button', { type: 'button', class: 'icon-button', 'aria-label': 'Close', onclick: () => els.settings.close() }, icon('x', 16))),
  h('div', { class: 'status-list' },
    statusItem('Ollama', sys?.ollama?.running, sys?.ollama?.running ? `${models.length} models installed` : 'Not running'),
    statusItem('GPU', Boolean(devices.gpu), devices.gpu ?? 'Not detected; using the CPU'),
    statusItem('Frame embeddings', true, `CLIP on ${(devices.clip ?? 'cpu').toUpperCase()}`),
    statusItem('FFmpeg', sys?.ffmpeg, sys?.ffmpeg ? 'Available' : 'Install FFmpeg and add it to PATH')),
  sys?.missing_models?.length ? h('p', { class: 'dialog-note' }, 'Not installed: ',
    sys.missing_models.map((name, i) => [i ? ', ' : '', h('code', {}, `ollama pull ${name}`)])) : null,
  h('div', { class: 'fields' },
    field('planner', 'Query planner', 'Turns your question into searchable evidence.', 'models-text'),
    field('verifier', 'Verifier', 'Second, stricter check of each match.', 'models-vision'),
    field('vision', 'Scene and vision model', 'Describes scenes during indexing, and checks candidates, boundaries, and on-screen text. Changing it requires re-indexing.', 'models-vision', true),
    h('label', { class: 'field field-wide' },
      h('span', { class: 'field-label field-label-row' }, 'Frames per vision call', framesOut),
      frames,
      h('small', { class: 'field-help' }, 'More frames catch brief actions but slow each vision call. Re-index after changing.'))),
  options('models-text', 'completion'),
  options('models-vision', 'vision'),
  h('div', { class: 'dialog-actions' },
    h('button', { type: 'button', class: 'button button-ghost', onclick: () => els.settings.close() }, 'Cancel'),
    h('button', { type: 'submit', class: 'button button-primary' }, 'Save')));
  els.settings.replaceChildren(form);
  els.settings.showModal();
}

// ------------------------------------------------------------------ misc

function updateHash() {
  const params = new URLSearchParams();
  if (state.selectedId) params.set('v', state.selectedId);
  if (state.searchId) params.set('s', state.searchId);
  const hash = params.toString();
  history.replaceState(null, '', hash ? `#${hash}` : location.pathname);
}

function tickElapsed() {
  for (const el of document.querySelectorAll('.elapsed[data-since]')) {
    const since = Number(el.dataset.since);
    el.textContent = since ? formatElapsed(Date.now() / 1000 - since) : '';
  }
}

function bindEvents() {
  els.addVideo.addEventListener('click', () => els.fileInput.click());
  els.emptyChoose.addEventListener('click', () => els.fileInput.click());
  els.fileInput.addEventListener('change', () => {
    addFiles([...els.fileInput.files]);
    els.fileInput.value = '';
  });
  els.systemPill.addEventListener('click', openSettings);
  els.form.addEventListener('submit', (event) => {
    event.preventDefault();
    const query = els.input.value.trim();
    if (query) runSearch(query);
    else els.input.focus();
  });
  for (const button of els.modeButtons) button.addEventListener('click', () => setMode(button.dataset.mode));
  els.optConfidence.addEventListener('input', () => { els.optConfidenceOut.textContent = pct(els.optConfidence.value); });
  els.optCandidates.addEventListener('input', () => { els.optCandidatesOut.textContent = els.optCandidates.value; });
  document.addEventListener('click', (event) => {
    if (els.options.open && !els.options.contains(event.target)) els.options.open = false;
  });
  for (const tab of els.tabs) {
    tab.addEventListener('click', () => {
      state.tab = tab.dataset.tab;
      rendered.tab = null;
      renderTabs();
    });
  }
  document.addEventListener('keydown', (event) => {
    const typing = event.target.closest('input, textarea, select, [contenteditable="true"]');
    if (event.key === '/' && !typing && !els.input.disabled && !els.workspace.hidden) {
      event.preventDefault();
      els.input.focus();
    }
    if (event.key === 'Escape' && els.options.open) els.options.open = false;
  });

  const hasFiles = (event) => [...(event.dataTransfer?.types || [])].includes('Files');
  let dragDepth = 0;
  window.addEventListener('dragenter', (event) => {
    if (!hasFiles(event)) return;
    event.preventDefault();
    dragDepth += 1;
    els.dropOverlay.hidden = false;
  });
  window.addEventListener('dragover', (event) => { if (hasFiles(event)) event.preventDefault(); });
  window.addEventListener('dragleave', (event) => {
    if (!hasFiles(event)) return;
    dragDepth = Math.max(0, dragDepth - 1);
    if (!dragDepth) els.dropOverlay.hidden = true;
  });
  window.addEventListener('drop', (event) => {
    if (!hasFiles(event)) return;
    event.preventDefault();
    dragDepth = 0;
    els.dropOverlay.hidden = true;
    addFiles([...event.dataTransfer.files]);
  });

  els.video.addEventListener('timeupdate', onTimeUpdate);
  els.video.addEventListener('play', () => requestAnimationFrame(frameLoop));
  els.video.addEventListener('seeking', () => {
    if (!programmaticSeek) state.segmentEnd = null;
    programmaticSeek = false;
  });
}

async function init() {
  hydrateIcons();
  bindEvents();
  setMode(state.mode);
  const params = new URLSearchParams(location.hash.slice(1));
  try {
    await Promise.all([loadSystem(), loadVideos()]);
  } catch {
    state.offline = true;
  }
  const requested = params.get('v');
  const id = state.videos.some((v) => v.id === requested) ? requested : state.videos[0]?.id ?? null;
  await selectVideo(id, id === requested ? params.get('s') : null);
  schedulePoll();
  setInterval(loadSystem, 20000);
  setInterval(tickElapsed, 1000);
}

init();
