import assert from 'node:assert/strict';
import { test } from 'node:test';
import { normalizeName } from '../frontend/cats.mjs';
import { AdoptionConflict, STORAGE_KEY, createApiStorage, createLocalStorage } from '../frontend/storage.mjs';

function memoryStorage() {
  const values = new Map();
  return { getItem: key => values.get(key) ?? null, setItem: (key, value) => values.set(key, value) };
}

test('imię obsługuje polskie znaki i normalizuje odstępy, odrzuca nieprawidłowe dane', () => {
  assert.equal(normalizeName('  Żółta   Łapka  '), 'Żółta Łapka');
  for (const value of [null, '', 'A', '<script>', 'x'.repeat(25)]) assert.throws(() => normalizeName(value));
});

test('nowy adapter po odświeżeniu odczytuje adopcję i jej imię', async () => {
  const memory = memoryStorage();
  const events = [];
  const firstPage = createLocalStorage(() => memory, event => events.push(event));
  assert.deepEqual(await firstPage.load(), {});
  const saved = await firstPage.adopt('miso', 'Rudy');
  const refreshedPage = createLocalStorage(() => memory);
  assert.deepEqual(await refreshedPage.load(), { miso: saved });
  assert.equal(saved.name, 'Rudy');
  assert.ok(events.some(event => event.text === 'localStorage.setItem()'));
  await assert.rejects(() => refreshedPage.adopt('miso', 'Inny'), AdoptionConflict);
  assert.equal((await refreshedPage.load()).miso.name, 'Rudy');
});

test('nieznany kot ani uszkodzone dane nie nadpisują istniejącego zapisu', async () => {
  const memory = memoryStorage();
  const storage = createLocalStorage(() => memory);
  await assert.rejects(() => storage.adopt('nieznany', 'Rudy'));
  assert.equal(memory.getItem(STORAGE_KEY), null);
  memory.setItem(`${STORAGE_KEY}.miso`, '{broken');
  await assert.rejects(() => storage.load(), /Nie można odczytać/);
  await assert.rejects(() => storage.adopt('miso', 'Rudy'));
  assert.equal(memory.getItem(`${STORAGE_KEY}.miso`), '{broken');
  memory.setItem(`${STORAGE_KEY}.miso`, JSON.stringify({ name: 'Rudy', adoptedAt: 'invalid' }));
  await assert.rejects(() => storage.load(), /Nie można odczytać/);
});

test('zapis innego kota w drugiej karcie nie usuwa potwierdzonej adopcji', async () => {
  const memory = memoryStorage();
  const originalRead = memory.getItem;
  const first = createLocalStorage(() => memory);
  const second = createLocalStorage(() => memory);
  let interleaved;
  let triggered = false;
  memory.getItem = key => {
    const value = originalRead(key);
    if (!triggered && key === `${STORAGE_KEY}.miso`) {
      triggered = true;
      interleaved = second.adopt('luna', 'Figa');
    }
    return value;
  };
  await first.adopt('miso', 'Rudy');
  await interleaved;
  const saved = await createLocalStorage(() => memory).load();
  assert.equal(saved.miso.name, 'Rudy');
  assert.equal(saved.luna.name, 'Figa');
});

test('blokada przeglądarki jest błędem, bez pozornego sukcesu', async () => {
  const memory = memoryStorage();
  memory.setItem = () => { throw new Error('QuotaExceededError'); };
  const events = [];
  const storage = createLocalStorage(() => memory, event => events.push(event));
  await assert.rejects(() => storage.adopt('luna', 'Figa'), /odmówiła zapisu/);
  assert.equal(events.filter(event => event.level === 'success').length, 0);
  await assert.rejects(() => createLocalStorage(() => { throw new Error('SecurityError'); }).load(), /Nie można odczytać/);
});

test('API wysyła imię na serwer i loguje potwierdzenie dopiero po odpowiedzi', async () => {
  const events = [];
  const record = { name: 'Figa', adoptedAt: '2026-11-16T12:00:00.000Z' };
  const storage = createApiStorage(async (path, options) => {
    assert.equal(path, '/api/adoptions');
    assert.equal(options.method, 'POST');
    assert.deepEqual(JSON.parse(options.body), { catId: 'luna', name: 'Figa' });
    assert.equal(events.some(event => event.text.includes('potwierdził zapis')), false);
    return new Response(JSON.stringify({ adoption: record }), { status: 201 });
  }, event => events.push(event));
  assert.deepEqual(await storage.adopt('luna', 'Figa'), record);
  assert.ok(events.some(event => event.text === 'PostgreSQL potwierdził zapis'));
});

test('awaria serwera nie przełącza zapisu na localStorage; konflikt zachowuje istniejące imię', async () => {
  const unavailable = createApiStorage(async () => { throw new TypeError('Failed to fetch'); });
  await assert.rejects(() => unavailable.load(), /Nie udało się połączyć/);
  const existing = { name: 'Figa', adoptedAt: '2026-11-16T12:00:00.000Z' };
  const conflict = createApiStorage(async () => new Response(JSON.stringify({ error: 'Konflikt', adoption: existing }), { status: 409 }));
  await assert.rejects(() => conflict.adopt('luna', 'Inna'), error => error instanceof AdoptionConflict && error.adoption.name === 'Figa');
});
