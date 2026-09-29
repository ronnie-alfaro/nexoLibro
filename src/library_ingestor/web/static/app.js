import {renderAnswer} from './answer-format.js';

const $ = (id) => document.getElementById(id);
const number = new Intl.NumberFormat('es');
const storageKey = 'nexolibro.conversations.v1';
let sessions = [];
try { sessions = JSON.parse(localStorage.getItem(storageKey) || '[]'); if (!Array.isArray(sessions)) sessions = []; } catch { sessions = []; }
let current = {id: crypto.randomUUID(), title: '', turns: [], scope: null};
let controller = null;
let catalogController = null;
let offset = 0;
let currentView = 'chat';
let searchTimer;
let assistantName = 'Asistente';

const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)');
function scrollBehavior() { return reducedMotion.matches ? 'instant' : 'smooth'; }

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}
function announce(text) { $('live-status').textContent = text; }
function notice(text) { $('service-notice').textContent = text; $('service-notice').hidden = !text; }
function save() {
  if (!current.turns.length) return;
  sessions = [structuredClone(current), ...sessions.filter((item) => item.id !== current.id)].slice(0, 10);
  try { localStorage.setItem(storageKey, JSON.stringify(sessions)); }
  catch { notice('No hay espacio para guardar más historial en este navegador. La consulta sigue disponible.'); }
  renderHistory();
}
function renderHistory() {
  const list = $('history-list'); list.replaceChildren();
  if (!sessions.length) list.append(element('p', 'empty-history', 'Tus consultas aparecerán aquí.'));
  for (const session of sessions) {
    const button = element('button', 'history-item', session.title || 'Consulta');
    button.title = session.title;
    button.addEventListener('click', () => {
      if (controller) return;
      current = structuredClone(session); showView('chat'); renderConversation();
    });
    list.append(button);
  }
}
function showView(view) {
  currentView = view;
  $('chat-view').hidden = view !== 'chat'; $('books-view').hidden = view !== 'books';
  $('nav-chat').classList.toggle('active', view === 'chat'); $('nav-books').classList.toggle('active', view === 'books');
  $('nav-chat').setAttribute('aria-current', view === 'chat' ? 'page' : 'false');
  $('nav-books').setAttribute('aria-current', view === 'books' ? 'page' : 'false');
  $('view-name').textContent = view === 'chat' ? 'Consultar' : 'Biblioteca';
  document.body.classList.remove('nav-open'); $('menu-toggle').setAttribute('aria-expanded', 'false');
  if (view === 'books') loadBooks();
}
function setScope() {
  $('scope-chip').hidden = !current.scope;
  $('scope-title').textContent = current.scope ? `Consultando: ${current.scope.title}` : '';
  $('composer-scope').textContent = current.scope ? 'Un libro seleccionado' : 'Toda la biblioteca';
}
function fresh(scope = null) {
  if (controller) { controller.abort(); return; }
  current = {id: crypto.randomUUID(), title: '', turns: [], scope};
  showView('chat'); renderConversation(); $('question').value = ''; $('question').focus();
}
function renderConversation() {
  const hasTurns = current.turns.length > 0;
  $('welcome').hidden = hasTurns;
  $('conversation').hidden = !hasTurns;
  $('chat-view').classList.toggle('in-conversation', hasTurns);
  $('conversation').replaceChildren();
  for (const turn of current.turns) {
    const parts = renderTurn(turn.question);
    parts.text.textContent = turn.answer;
    renderSources(parts.sources, turn.sources || []);
    decorateCitations(parts.text, turn.answer, parts.sources, turn.sources || []);
    finishTurn(parts, turn);
  }
  setScope();
}
function renderTurn(question) {
  const root = element('article', 'chat-turn');
  const user = element('div', 'user-question'); user.append(element('p', '', question)); root.append(user);
  const heading = element('div', 'answer-heading');
  const stage = element('span', 'answer-stage', 'Buscando…');
  heading.append(element('span', '', assistantName), stage); root.append(heading);
  const text = element('div', 'answer-text'); root.append(text);
  const sources = element('div'); root.append(sources);
  const warnings = element('div', 'answer-warning'); root.append(warnings);
  const meta = element('div', 'answer-meta'); root.append(meta);
  $('conversation').append(root);
  return {root, stage, text, sources, warnings, meta};
}
function renderSources(target, sources) {
  target.replaceChildren();
  if (!sources.length) return;
  const panel = element('details', 'source-list');
  panel.append(element('summary', '', `Explorar fuentes · ${sources.length} fragmentos`));
  const grid = element('div', 'source-grid');
  for (const source of sources) {
    const card = element('details', 'source-card'); card.dataset.number = String(source.number);
    const summary = element('summary');
    const title = element('div', 'source-name');
    title.append(element('span', 'source-number', `[${source.number}]`), element('span', '', source.title));
    const location = [source.author, source.section, source.page_number ? `p. ${source.page_number}` : null].filter(Boolean).join(' · ');
    summary.append(title, element('div', 'source-location', location || source.filename));
    card.append(summary, element('p', 'source-excerpt', source.text));
    grid.append(card);
  }
  panel.append(grid); target.append(panel);
}
function decorateCitations(target, text, sourcesNode, sources) {
  renderAnswer(target, text, sources, (value) => {
        const panel = sourcesNode.querySelector('.source-list'); if (panel) panel.open = true;
        const card = [...sourcesNode.querySelectorAll('.source-card')].find((node) => node.dataset.number === String(value));
        if (card) {
          sourcesNode.querySelectorAll('.highlight').forEach((node) => node.classList.remove('highlight'));
          card.open = true; card.classList.add('highlight'); card.scrollIntoView({behavior: scrollBehavior(), block: 'center'});
        }
  });
}
function finishTurn(parts, turn) {
  parts.stage.textContent = turn.stopped ? 'Interrumpida' : 'Consulta completada';
  parts.text.classList.remove('generating');
  parts.warnings.textContent = (turn.warnings || []).join(' ');
  parts.meta.replaceChildren();
  if (turn.seconds !== undefined) parts.meta.append(element('span', '', `${turn.seconds} s · respuesta local`));
  const copy = element('button', 'copy-answer', 'Copiar respuesta');
  copy.addEventListener('click', async () => {
    try { await navigator.clipboard.writeText(turn.answer); copy.textContent = 'Copiada'; }
    catch { copy.textContent = 'Selecciona el texto para copiar'; }
  });
  parts.meta.append(copy);
}
async function ask(event) {
  event.preventDefault();
  const question = $('question').value.trim(); if (!question || controller) return;
  const history = current.turns.filter((turn) => !turn.stopped).slice(-3).flatMap((turn) => [
    {role: 'user', content: turn.question.slice(0, 6000)}, {role: 'assistant', content: turn.answer.slice(0, 6000)}
  ]);
  controller = new AbortController();
  document.body.classList.add('is-generating');
  $('send-button').hidden = true; $('stop-button').hidden = false;
  $('question').disabled = true; $('new-chat').disabled = true;
  $('welcome').hidden = true;
  $('conversation').hidden = false; $('chat-view').classList.add('in-conversation');
  const parts = renderTurn(question); parts.text.classList.add('generating');
  const turn = {question, answer: '', sources: [], warnings: []};
  let completed = false; let paintScheduled = false; let lastPaint = 0;
  const paint = () => {
    // Coalesce fast token streams; finalization always renders the complete text.
    if (paintScheduled || performance.now() - lastPaint < 100) return;
    paintScheduled = true;
    requestAnimationFrame(() => {
      decorateCitations(parts.text, turn.answer, parts.sources, turn.sources);
      lastPaint = performance.now(); paintScheduled = false;
    });
  };
  const onEvent = (data) => {
    if (data.type === 'stage') { parts.stage.textContent = data.message; announce(data.message); }
    if (data.type === 'sources') { turn.sources = data.sources; renderSources(parts.sources, turn.sources); }
    if (data.type === 'token') { turn.answer += data.text; paint(); }
    if (data.type === 'error') throw new Error(data.message);
    if (data.type === 'done') { completed = true; Object.assign(turn, data); }
  };
  parts.root.scrollIntoView({behavior: scrollBehavior(), block: 'start'});
  try {
    const response = await fetch('/api/ask', {
      method: 'POST', headers: {'Content-Type': 'application/json', 'X-NexoLibro-Client': 'local'},
      body: JSON.stringify({question, history, document_id: current.scope?.document_id || null}), signal: controller.signal
    });
    if (!response.ok) {
      const error = await response.json();
      throw new Error(typeof error.detail === 'string' ? error.detail : 'No se pudo enviar la pregunta.');
    }
    const reader = response.body.getReader(); const decoder = new TextDecoder(); let buffer = '';
    while (true) {
      const {value, done} = await reader.read(); if (done) break;
      buffer += decoder.decode(value, {stream: true});
      let end;
      while ((end = buffer.indexOf('\n\n')) >= 0) {
        const frame = buffer.slice(0, end); buffer = buffer.slice(end + 2);
        const data = frame.split('\n').filter((line) => line.startsWith('data:')).map((line) => line.slice(5).trim()).join('\n');
        if (data) onEvent(JSON.parse(data));
      }
    }
    if (!completed) throw new Error('La conexión terminó antes de completar la respuesta.');
    if (!current.title) current.title = question;
    current.turns.push(turn); current.turns = current.turns.slice(-12); save();
    $('question').value = ''; announce('Respuesta completada. Puedes explorar sus fuentes.');
  } catch (error) {
    controller.abort();
    turn.stopped = true;
    turn.warnings = [error.name === 'AbortError' ? 'Consulta detenida.' : error.message];
    announce(turn.warnings[0]);
  } finally {
    // Wait for pending frame before replacing text nodes with citation buttons.
    await new Promise(requestAnimationFrame);
    decorateCitations(parts.text, turn.answer, parts.sources, turn.sources);
    finishTurn(parts, turn);
    controller = null; $('send-button').hidden = false; $('stop-button').hidden = true;
    document.body.classList.remove('is-generating');
    $('question').disabled = false; $('new-chat').disabled = false;
    $('question').focus({preventScroll: true});
  }
}
async function loadStatus() {
  try {
    const response = await fetch('/api/status'); if (!response.ok) throw new Error('No se pudo leer el estado local.');
    const status = await response.json();
    $('book-count').textContent = number.format(status.books); $('nav-book-count').textContent = number.format(status.books);
    $('chunk-count').textContent = number.format(status.chunks);
    $('connection-dot').classList.toggle('ready', status.model_ready);
    $('connection-text').textContent = status.model_ready ? 'Modelo disponible' : 'Modelo desconectado';
    $('connection-text').title = status.model || status.model_error || '';
    notice(status.index_error || (status.model_ready ? '' : status.model_error || 'Inicia el modelo local para consultar.'));
  } catch (error) { $('connection-text').textContent = 'Sin conexión'; notice(error.message); }
}
async function loadBooks() {
  if (catalogController) catalogController.abort(); catalogController = new AbortController();
  $('catalog-total').textContent = 'Buscando…';
  try {
    const response = await fetch(`/api/books?${new URLSearchParams({q: $('book-search').value, offset, limit: 24})}`, {signal: catalogController.signal});
    if (!response.ok) { const error = await response.json(); throw new Error(error.detail || 'No se pudo abrir la biblioteca.'); }
    const data = await response.json(); $('book-grid').replaceChildren();
    $('catalog-total').textContent = `${number.format(data.total)} libros`;
    if (!data.items.length) $('book-grid').append(element('p', 'catalog-empty', 'No hay libros completados que coincidan con esta búsqueda.'));
    for (const book of data.items) {
      const card = element('button', 'book-card');
      card.append(element('span', 'book-format', book.file_type.toUpperCase() + (book.language ? ` · ${book.language}` : '')),
        element('h3', '', book.title), element('p', '', book.author || 'Autor no indicado'), element('small', '', `${number.format(book.chunks)} fragmentos · Consultar ↗`));
      card.addEventListener('click', () => { if (!controller) fresh({document_id: book.document_id, title: book.title}); });
      $('book-grid').append(card);
    }
    $('previous-page').disabled = offset === 0; $('next-page').disabled = offset + 24 >= data.total;
    $('page-info').textContent = data.total ? `${offset + 1}–${Math.min(offset + 24, data.total)} de ${number.format(data.total)}` : 'Sin resultados';
  } catch (error) {
    if (error.name === 'AbortError') return;
    $('book-grid').replaceChildren(element('p', 'catalog-empty', error.message)); $('catalog-total').textContent = 'No disponible';
  }
}
$('ask-form').addEventListener('submit', ask);
$('question').addEventListener('keydown', (event) => { if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) { event.preventDefault(); $('ask-form').requestSubmit(); } });
$('stop-button').addEventListener('click', () => controller?.abort());
$('new-chat').addEventListener('click', () => fresh());
$('nav-chat').addEventListener('click', () => showView('chat'));
$('nav-books').addEventListener('click', () => showView('books'));
$('clear-scope').addEventListener('click', () => { if (!controller) fresh(); });
$('book-search').addEventListener('input', () => { clearTimeout(searchTimer); searchTimer = setTimeout(() => { offset = 0; loadBooks(); }, 250); });
$('previous-page').addEventListener('click', () => { offset = Math.max(0, offset - 24); loadBooks(); });
$('next-page').addEventListener('click', () => { offset += 24; loadBooks(); });
$('clear-history').addEventListener('click', () => {
  if (controller) return;
  sessions = []; try { localStorage.removeItem(storageKey); } catch { /* private browsing */ }
  fresh(); renderHistory(); announce('Historial local borrado.');
});
$('help-button').addEventListener('click', () => $('help-dialog').showModal());
$('close-help').addEventListener('click', () => $('help-dialog').close());
$('help-dialog').addEventListener('click', (event) => { if (event.target === $('help-dialog')) $('help-dialog').close(); });
$('menu-toggle').addEventListener('click', () => { const open = document.body.classList.toggle('nav-open'); $('menu-toggle').setAttribute('aria-expanded', String(open)); });
document.addEventListener('keydown', (event) => { if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 'k') { event.preventDefault(); if (!controller) fresh(); } });
async function loadBranding() {
  try {
    const branding = await (await fetch('/api/config')).json();
    $('brand-name').firstChild.textContent = branding.name;
    $('brand-tagline').textContent = branding.tagline;
    $('welcome-title').textContent = branding.welcome_title;
    $('welcome-description').textContent = branding.welcome_description;
    document.title = `${branding.name} — ${branding.tagline}`;
    assistantName = branding.assistant_name;
    document.querySelectorAll('.answer-heading > span:first-child').forEach((node) => { node.textContent = assistantName; });
  } catch { /* Defaults in HTML keep the interface usable while starting. */ }
}
renderHistory(); setScope(); loadBranding(); loadStatus();
setInterval(() => { if (!document.hidden && !controller) loadStatus(); }, 30000);
