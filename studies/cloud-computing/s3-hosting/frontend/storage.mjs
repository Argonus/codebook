import { cats, validateAdoption } from './cats.mjs';

export const STORAGE_KEY = 'koci-azyl.adoptions.v1';

export class AdoptionConflict extends Error {
  constructor(adoption) {
    super('Ten kot znalazł już dom. Odświeżyliśmy jego historię.');
    this.adoption = adoption;
  }
}

function checkRecords(records) {
  if (!records || typeof records !== 'object' || Array.isArray(records)) throw new Error('Nieprawidłowe dane adopcji.');
  for (const [id, record] of Object.entries(records)) {
    if (!record || validateAdoption(id, record.name) !== record.name || typeof record.adoptedAt !== 'string' || !Number.isFinite(Date.parse(record.adoptedAt))) {
      throw new Error('Nieprawidłowe dane adopcji.');
    }
  }
  return records;
}

export function createLocalStorage(getStorage, onEvent = () => {}) {
  function readCat(catId) {
    const raw = getStorage().getItem(`${STORAGE_KEY}.${catId}`);
    if (raw === null) return undefined;
    return checkRecords({ [catId]: JSON.parse(raw) })[catId];
  }
  return {
    async load() {
      onEvent({ text: 'localStorage.getItem()', detail: `${STORAGE_KEY}.*` });
      try {
        const records = {};
        for (const cat of cats) {
          const record = readCat(cat.id);
          if (record) records[cat.id] = record;
        }
        onEvent({ text: `Odczytano adopcje: ${Object.keys(records).length}`, detail: 'Dane z tej przeglądarki · bez żądania HTTP', level: 'success' });
        return records;
      } catch {
        throw new Error('Nie można odczytać localStorage. Sprawdź uprawnienia przeglądarki lub zapisane dane.');
      }
    },
    async adopt(catId, value) {
      const name = validateAdoption(catId, value);
      const existing = readCat(catId);
      if (existing) throw new AdoptionConflict(existing);
      const adoption = { name, adoptedAt: new Date().toISOString() };
      onEvent({ text: 'localStorage.setItem()', detail: `${catId} → ${name}` });
      try {
        getStorage().setItem(`${STORAGE_KEY}.${catId}`, JSON.stringify(adoption));
      } catch {
        throw new Error('Przeglądarka odmówiła zapisu. Adopcja nie została zapisana.');
      }
      onEvent({ text: 'Zapis zakończony', detail: 'Tylko ta przeglądarka · 0 żądań do API', level: 'success' });
      return adoption;
    },
  };
}

export function createApiStorage(fetcher, onEvent = () => {}) {
  async function request(path, options) {
    const method = options?.method ?? 'GET';
    onEvent({ text: `${method} ${path}`, detail: 'Przeglądarka → serwer HTTP → PostgreSQL' });
    let response;
    try {
      response = await fetcher(path, { ...options, signal: AbortSignal.timeout(10000), cache: 'no-store' });
    } catch {
      throw new Error('Nie udało się połączyć z serwerem. Sprawdź połączenie i spróbuj ponownie.');
    }
    const body = await response.json();
    onEvent({ text: `HTTP ${response.status}`, detail: response.ok ? 'Odpowiedź z serwera' : body.error, level: response.ok ? 'success' : 'error' });
    if (response.status === 409 && body.adoption) {
      checkRecords({ [options.catId]: body.adoption });
      throw new AdoptionConflict(body.adoption);
    }
    if (!response.ok) throw new Error(body.error || 'Serwer nie mógł wykonać operacji.');
    return body;
  }
  return {
    async load() {
      const body = await request('/api/adoptions');
      const records = checkRecords(body.adoptions);
      onEvent({ text: `PostgreSQL: odczytano adopcje: ${Object.keys(records).length}`, detail: 'Stan wspólny dla wszystkich odwiedzających', level: 'success' });
      return records;
    },
    async adopt(catId, value) {
      const name = validateAdoption(catId, value);
      const body = await request('/api/adoptions', {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ catId, name }), catId,
      });
      checkRecords({ [catId]: body.adoption });
      onEvent({ text: 'PostgreSQL potwierdził zapis', detail: `${catId} → ${body.adoption.name} · trwały wolumen Dockera`, level: 'success' });
      return body.adoption;
    },
  };
}
