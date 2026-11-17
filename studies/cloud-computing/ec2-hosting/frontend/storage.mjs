import { checkBatch, createBatch, initialBatch, normalizeName, validateAdoption } from './cats.mjs';

export const STORAGE_KEY = 'koci-azyl.adoptions.v1';
export const BATCH_KEY = `${STORAGE_KEY}.batch`;

export class AdoptionConflict extends Error {
  constructor(adoption) {
    super('Ten kot znalazł już dom. Odświeżyliśmy jego historię.');
    this.adoption = adoption;
  }
}

export class BatchConflict extends Error {}

function checkAdoption(record) {
  if (!record || normalizeName(record.name) !== record.name || typeof record.adoptedAt !== 'string' || !Number.isFinite(Date.parse(record.adoptedAt))) throw new Error('Nieprawidłowe dane adopcji.');
  return record;
}

function checkState(state) {
  const batch = checkBatch(state);
  if (!state.adoptions || typeof state.adoptions !== 'object' || Array.isArray(state.adoptions)) throw new Error('Nieprawidłowe dane adopcji.');
  for (const [id, record] of Object.entries(state.adoptions)) {
    validateAdoption(id, record.name, batch.cats);
    checkAdoption(record);
  }
  return { ...batch, adoptions: state.adoptions };
}

export function createLocalStorage(getStorage, onEvent = () => {}) {
  function readBatch() {
    const raw = getStorage().getItem(BATCH_KEY);
    return raw === null ? initialBatch : checkBatch(JSON.parse(raw));
  }
  function readCat(catId) {
    const raw = getStorage().getItem(`${STORAGE_KEY}.${catId}`);
    return raw === null ? undefined : checkAdoption(JSON.parse(raw));
  }
  function readState() {
    const batch = readBatch();
    const adoptions = {};
    for (const cat of batch.cats) {
      const record = readCat(cat.id);
      if (record) adoptions[cat.id] = record;
    }
    return { ...batch, adoptions };
  }
  function write(key, value) {
    try { getStorage().setItem(key, JSON.stringify(value)); }
    catch { throw new Error('Przeglądarka odmówiła zapisu. Zmiana nie została zapisana.'); }
  }
  return {
    async load() {
      onEvent({ text: 'localStorage.getItem()', detail: `${STORAGE_KEY}.*` });
      try {
        const state = readState();
        onEvent({ text: `Odczytano adopcje: ${Object.keys(state.adoptions).length}`, detail: 'Aktualna trójka z tej przeglądarki · bez HTTP', level: 'success' });
        return state;
      } catch {
        throw new Error('Nie można odczytać localStorage. Sprawdź uprawnienia przeglądarki lub zapisane dane.');
      }
    },
    async adopt(catId, value) {
      const batch = readBatch();
      const name = validateAdoption(catId, value, batch.cats);
      const existing = readCat(catId);
      if (existing) throw new AdoptionConflict(existing);
      const adoption = { name, adoptedAt: new Date().toISOString() };
      onEvent({ text: 'localStorage.setItem()', detail: `${catId} → ${name}` });
      write(`${STORAGE_KEY}.${catId}`, adoption);
      onEvent({ text: 'Zapis zakończony', detail: 'Tylko ta przeglądarka · 0 żądań do API', level: 'success' });
      return adoption;
    },
    async draw(batchId) {
      const state = readState();
      if (state.batchId !== batchId) throw new BatchConflict('Zestaw kotów już się zmienił. Odświeżamy stronę.');
      if (!state.cats.every(cat => state.adoptions[cat.id])) throw new Error('Najpierw adoptuj wszystkie trzy koty.');
      const next = createBatch(state.cats);
      if (readBatch().batchId !== batchId) throw new BatchConflict('Zestaw kotów już się zmienił. Odświeżamy stronę.');
      onEvent({ text: 'localStorage.setItem()', detail: 'Losowanie trzech nowych kotów' });
      write(BATCH_KEY, next);
      onEvent({ text: 'Nowa trójka zapisana', detail: 'Poprzednie adopcje zachowane w localStorage', level: 'success' });
      return { ...next, adoptions: {} };
    },
  };
}

export function createApiStorage(fetcher, onEvent = () => {}) {
  async function request(path, options) {
    onEvent({ text: `${options?.method ?? 'GET'} ${path}`, detail: 'Przeglądarka → HTTP API → PostgreSQL' });
    let response;
    try { response = await fetcher(path, { ...options, signal: AbortSignal.timeout(10000), cache: 'no-store' }); }
    catch { throw new Error('Nie udało się połączyć z serwerem. Sprawdź połączenie i spróbuj ponownie.'); }
    const body = await response.json();
    onEvent({ text: `HTTP ${response.status}`, detail: response.ok ? 'Odpowiedź z serwera' : body.error, level: response.ok ? 'success' : 'error' });
    if (response.status === 409) {
      if (body.adoption) throw new AdoptionConflict(checkAdoption(body.adoption));
      throw new BatchConflict(body.error);
    }
    if (!response.ok) throw new Error(body.error || 'Serwer nie mógł wykonać operacji.');
    return body;
  }
  function post(body) { return { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) }; }
  return {
    async load() {
      const state = checkState(await request('/api/cats'));
      onEvent({ text: `PostgreSQL: odczytano adopcje: ${Object.keys(state.adoptions).length}`, detail: 'Aktualna trójka wspólna dla wszystkich odwiedzających', level: 'success' });
      return state;
    },
    async adopt(catId, value) {
      const body = await request('/api/adoptions', post({ catId, name: normalizeName(value) }));
      const adoption = checkAdoption(body.adoption);
      onEvent({ text: 'PostgreSQL potwierdził zapis', detail: `${catId} → ${adoption.name} · trwały wolumen Dockera`, level: 'success' });
      return adoption;
    },
    async draw(batchId) {
      const state = checkState(await request('/api/cats/draw', post({ batchId })));
      onEvent({ text: 'PostgreSQL: nowa trójka zapisana', detail: 'Poprzednie adopcje zachowane w bazie', level: 'success' });
      return state;
    },
  };
}
