import { config } from './config.mjs';
import { initialBatch } from './dogs.mjs';
import { dogDrawing } from './illustrations.mjs';
import { AdoptionConflict, BatchConflict, STORAGE_KEY, createApiStorage, createLocalStorage } from './storage.mjs';

const $ = selector => document.querySelector(selector);
const cards = new Map();
let adoptions = {};
let ready = false;
let current = initialBatch;
let drawing = false;

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
  for (const dog of current.cats) {
    const card = cards.get(dog.id);
    const adoption = adoptions[dog.id];
    const busy = card.dataset.saving === 'true';
    card.classList.toggle('is-adopted', Boolean(adoption));
    card.querySelector('.dog-name').textContent = adoption?.name ?? dog.name;
    card.querySelector('.dog-status').textContent = adoption ? 'Mam dom ♡' : 'Czekam w klatce';
    const input = card.querySelector('input');
    if (adoption) input.value = adoption.name;
    input.disabled = !ready || Boolean(adoption) || busy;
    const button = card.querySelector('button');
    button.disabled = input.disabled;
    button.textContent = adoption ? 'Adoptowany' : busy ? 'Zapis…' : 'Adoptuj';
  }
  const remaining = current.cats.filter(dog => !adoptions[dog.id]).length;
  $('#draw-hint').textContent = !ready ? 'Wczytywanie stanu psów…' : remaining ? `Do adopcji pozostało: ${remaining}.` : 'Cała trójka ma już dom!';
  $('#draw').disabled = !ready || drawing || remaining !== 0;
  $('#draw').textContent = drawing ? 'Losuję…' : 'Losuj nowe psy';
}

function showState(state) {
  const changed = current.batchId !== state.batchId || cards.size === 0;
  current = { batchId: state.batchId, cats: state.cats };
  adoptions = state.adoptions;
  if (changed) {
    cards.clear();
    $('#dogs').replaceChildren();
    for (const dog of current.cats) renderDog(dog);
  }
}

async function load() {
  $('#page-error').hidden = true;
  try {
    showState(await storage.load());
    ready = true;
  } catch (error) {
    ready = false;
    $('#page-error span').textContent = error.message;
    $('#page-error').hidden = false;
    log({ text: 'Odczyt nie powiódł się', detail: error.message, level: 'error' });
  }
  update();
}

async function adopt(event, dog, card) {
  event.preventDefault();
  if (!ready || card.dataset.saving === 'true' || adoptions[dog.id]) return;
  const input = card.querySelector('input');
  const name = input.value;
  const batchId = current.batchId;
  const errorLabel = card.querySelector('.card-error');
  errorLabel.textContent = '';
  card.dataset.saving = 'true';
  update();
  try {
    const adoption = await storage.adopt(dog.id, name);
    if (current.batchId === batchId) adoptions[dog.id] = adoption;
    else await load();
    $('#announcement').textContent = `${adoption.name} wychodzi z klatki. Adopcja zapisana.`;
    log({ text: 'Drzwiczki otwarte. Pies wychodzi z klatki.', detail: 'Zapis został potwierdzony', level: 'success' });
  } catch (error) {
    if (error instanceof AdoptionConflict) adoptions[dog.id] = error.adoption;
    if (error instanceof BatchConflict) await load();
    errorLabel.textContent = error.message;
    log({ text: 'Adopcja nie została zapisana', detail: error.message, level: 'error' });
  } finally {
    card.dataset.saving = 'false';
    update();
  }
}

function renderDog(dog) {
  const card = document.createElement('article');
  card.className = 'dog-card';
  card.dataset.dog = dog.id;
  card.innerHTML = `<div class="dog-picture">${dogDrawing(dog.appearance)}</div><div class="dog-info"><div class="dog-heading"><h2 class="dog-name"></h2><span class="dog-status"></span></div><form><label for="name-${dog.id}">Nadaj psu imię</label><div class="name-row"><input id="name-${dog.id}" name="name" placeholder="Nowe imię" required minlength="2" maxlength="24" autocomplete="off" aria-describedby="error-${dog.id}" disabled><button type="submit" disabled>Adoptuj</button></div><p class="card-error" id="error-${dog.id}" role="alert"></p></form></div>`;
  card.querySelector('form').addEventListener('submit', event => adopt(event, dog, card));
  cards.set(dog.id, card);
  $('#dogs').append(card);
}
$('#draw').addEventListener('click', async () => {
  if (!ready || drawing || !current.cats.every(dog => adoptions[dog.id])) return;
  drawing = true;
  $('#page-error').hidden = true;
  update();
  try {
    showState(await storage.draw(current.batchId));
    $('#announcement').textContent = 'Wylosowano trzy nowe psy. Wszystkie czekają w klatkach.';
    log({ text: 'Trzy nowe psy czekają na adopcję', detail: 'Aktualny zestaw będzie zachowany po odświeżeniu', level: 'success' });
  } catch (error) {
    if (error instanceof BatchConflict) await load();
    $('#page-error span').textContent = error.message;
    $('#page-error').hidden = false;
    log({ text: 'Losowanie nie powiodło się', detail: error.message, level: 'error' });
  } finally {
    drawing = false;
    update();
  }
});
showState({ ...initialBatch, adoptions: {} });
update();
$('#retry').addEventListener('click', load);
if (!isServer) window.addEventListener('storage', event => { if (event.key?.startsWith(`${STORAGE_KEY}.`) || event.key === null) void load(); });
log({ text: 'Strona uruchomiona', detail: isServer ? 'Wersja EC2 · zapis na serwerze' : 'Wersja S3 · zapis lokalny' });
await load();
