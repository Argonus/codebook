import { config } from './config.mjs';
import { cats } from './cats.mjs';
import { catDrawing } from './illustrations.mjs';
import { AdoptionConflict, STORAGE_KEY, createApiStorage, createLocalStorage } from './storage.mjs';

const $ = selector => document.querySelector(selector);
const cards = new Map();
let adoptions = {};
let ready = false;

function log({ text, detail = '', level = 'info' }) {
  const line = document.createElement('li');
  line.className = `log-${level}`;
  const time = document.createElement('time');
  time.textContent = new Date().toLocaleTimeString('pl-PL', { hour12: false });
  const message = document.createElement('span');
  message.className = 'log-message';
  message.textContent = text;
  const extra = document.createElement('span');
  extra.className = 'log-detail';
  extra.textContent = detail ? ` — ${detail}` : '';
  line.append(time, message, extra);
  $('#logs').append(line);
  while ($('#logs').children.length > 30) $('#logs').firstElementChild.remove();
  $('#logs').scrollTop = $('#logs').scrollHeight;
}

const isServer = config.mode === 'ec2';
const storage = isServer ? createApiStorage(window.fetch.bind(window), log) : createLocalStorage(() => window.localStorage, log);
$('#version').textContent = isServer ? 'EC2 · PostgreSQL w Dockerze' : 'S3 · localStorage';
$('#route').textContent = isServer ? 'Przeglądarka → HTTP API → PostgreSQL' : 'Przeglądarka → localStorage (bez API)';
$('#persistence').textContent = isServer ? 'Adopcje zapisane na serwerze są wspólne dla wszystkich odwiedzających.' : 'Adopcje zapisują się tylko w tej przeglądarce, na tym urządzeniu.';
const authorName = document.createElement('strong');
authorName.textContent = `${config.author.firstName} ${config.author.lastName}`;
$('#author').append(authorName, `Grupa ${config.author.group} · nr albumu ${config.author.studentId}`);

function update() {
  for (const cat of cats) {
    const card = cards.get(cat.id);
    const adoption = adoptions[cat.id];
    const busy = card.dataset.saving === 'true';
    card.classList.toggle('is-adopted', Boolean(adoption));
    card.querySelector('.cat-name').textContent = adoption?.name ?? cat.name;
    card.querySelector('.cat-status').textContent = adoption ? 'Mam dom ♡' : 'Czekam w klatce';
    const input = card.querySelector('input');
    if (adoption) input.value = adoption.name;
    input.disabled = !ready || Boolean(adoption) || busy;
    const button = card.querySelector('button');
    button.disabled = input.disabled;
    button.textContent = adoption ? 'Adoptowany' : busy ? 'Zapis…' : 'Adoptuj';
  }
}

async function load() {
  $('#page-error').hidden = true;
  try {
    adoptions = await storage.load();
    ready = true;
  } catch (error) {
    ready = false;
    $('#page-error span').textContent = error.message;
    $('#page-error').hidden = false;
    log({ text: 'Odczyt nie powiódł się', detail: error.message, level: 'error' });
  }
  update();
}

async function adopt(event, cat, card) {
  event.preventDefault();
  if (!ready || card.dataset.saving === 'true' || adoptions[cat.id]) return;
  const input = card.querySelector('input');
  const name = input.value;
  const errorLabel = card.querySelector('.card-error');
  errorLabel.textContent = '';
  card.dataset.saving = 'true';
  update();
  try {
    adoptions[cat.id] = await storage.adopt(cat.id, name);
    $('#announcement').textContent = `${adoptions[cat.id].name} wychodzi z klatki. Adopcja zapisana.`;
    log({ text: 'Drzwiczki otwarte. Kot wychodzi z klatki.', detail: 'Zapis został potwierdzony', level: 'success' });
  } catch (error) {
    if (error instanceof AdoptionConflict) adoptions[cat.id] = error.adoption;
    errorLabel.textContent = error.message;
    log({ text: 'Adopcja nie została zapisana', detail: error.message, level: 'error' });
  } finally {
    card.dataset.saving = 'false';
    update();
  }
}

for (const cat of cats) {
  const card = document.createElement('article');
  card.className = 'cat-card';
  card.dataset.cat = cat.id;
  card.innerHTML = `<div class="cat-picture">${catDrawing(cat.id)}</div><div class="cat-info"><div class="cat-heading"><h2 class="cat-name"></h2><span class="cat-status"></span></div><form><label for="name-${cat.id}">Nadaj kotu imię</label><div class="name-row"><input id="name-${cat.id}" name="name" placeholder="Nowe imię" required minlength="2" maxlength="24" autocomplete="off" aria-describedby="error-${cat.id}" disabled><button type="submit" disabled>Adoptuj</button></div><p class="card-error" id="error-${cat.id}" role="alert"></p></form></div>`;
  card.querySelector('form').addEventListener('submit', event => adopt(event, cat, card));
  cards.set(cat.id, card);
  $('#cats').append(card);
}
update();
$('#retry').addEventListener('click', load);
if (!isServer) window.addEventListener('storage', event => { if (event.key?.startsWith(`${STORAGE_KEY}.`) || event.key === null) void load(); });
log({ text: 'Strona uruchomiona', detail: isServer ? 'Wersja EC2 · zapis na serwerze' : 'Wersja S3 · zapis lokalny' });
await load();
