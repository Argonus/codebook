import assert from 'node:assert/strict';
import { test } from 'node:test';
import { cats, createBatch, initialBatch, normalizeName } from '../frontend/cats.mjs';
import { AdoptionConflict, BatchConflict, BATCH_KEY, STORAGE_KEY, createApiStorage, createLocalStorage } from '../frontend/storage.mjs';

function memoryStorage() {
  const values = new Map();
  return { getItem: key => values.get(key) ?? null, setItem: (key, value) => values.set(key, value) };
}

test('imię obsługuje polskie znaki i odrzuca nieprawidłowe dane', () => {
  assert.equal(normalizeName('  Żółta   Łapka  '), 'Żółta Łapka');
  for (const value of [null, '', 'A', '<script>', 'x'.repeat(25)]) assert.throws(() => normalizeName(value));
});

test('nowy adapter odczytuje dotychczasowe adopcje bez zmiany formatu starych wpisów', async () => {
  const memory = memoryStorage();
  const legacy = { name: 'Rudy', adoptedAt: '2026-11-16T12:00:00.000Z' };
  memory.setItem(`${STORAGE_KEY}.miso`, JSON.stringify(legacy));
  const storage = createLocalStorage(() => memory);
  const state = await storage.load();
  assert.equal(state.batchId, 'initial');
  assert.deepEqual(state.adoptions, { miso: legacy });
  await assert.rejects(() => storage.adopt('miso', 'Inny'), AdoptionConflict);
  assert.equal((await storage.load()).adoptions.miso.name, 'Rudy');
});

test('losowanie wymaga adopcji trzech kotów, zachowuje historię i przetrwa odświeżenie', async () => {
  const memory = memoryStorage();
  const storage = createLocalStorage(() => memory);
  await assert.rejects(() => storage.draw('initial'), /Najpierw adoptuj/);
  assert.equal(memory.getItem(BATCH_KEY), null);
  for (const cat of cats) await storage.adopt(cat.id, cat.name);
  const history = cats.map(cat => memory.getItem(`${STORAGE_KEY}.${cat.id}`));
  const next = await storage.draw('initial');
  assert.notEqual(next.batchId, 'initial');
  assert.equal(new Set(next.cats.map(cat => cat.id)).size, 3);
  assert.equal(new Set(next.cats.map(cat => cat.name)).size, 3);
  assert.equal(new Set(next.cats.map(cat => cat.appearance)).size, 3);
  assert.ok(next.cats.every(cat => !cats.some(old => old.id === cat.id || old.name === cat.name)));
  assert.deepEqual(next.adoptions, {});
  assert.deepEqual(cats.map(cat => memory.getItem(`${STORAGE_KEY}.${cat.id}`)), history);
  await assert.rejects(() => storage.draw('initial'), BatchConflict);
  const refreshed = createLocalStorage(() => memory);
  assert.deepEqual(await refreshed.load(), next);
  await refreshed.adopt(next.cats[0].id, 'Figa');
  assert.equal((await createLocalStorage(() => memory).load()).adoptions[next.cats[0].id].name, 'Figa');
});

test('odmowa zapisu nowej trójki zachowuje poprzedni zestaw i jego adopcje', async () => {
  const memory = memoryStorage();
  const storage = createLocalStorage(() => memory);
  for (const cat of cats) await storage.adopt(cat.id, cat.name);
  const previous = await storage.load();
  memory.setItem = () => { throw new Error('QuotaExceededError'); };
  await assert.rejects(() => storage.draw('initial'), /odmówiła zapisu/);
  assert.deepEqual(await storage.load(), previous);
});

test('nieznany kot i uszkodzony zapis nie powodują nadpisania danych', async () => {
  const memory = memoryStorage();
  const storage = createLocalStorage(() => memory);
  await assert.rejects(() => storage.adopt('nieznany', 'Rudy'));
  memory.setItem(`${STORAGE_KEY}.miso`, '{broken');
  await assert.rejects(() => storage.load(), /Nie można odczytać/);
  await assert.rejects(() => storage.adopt('miso', 'Rudy'));
  assert.equal(memory.getItem(`${STORAGE_KEY}.miso`), '{broken');
  memory.setItem(BATCH_KEY, JSON.stringify({ batchId: 'bad', cats: [{ id: '<script>' }] }));
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
    if (!triggered && key === `${STORAGE_KEY}.miso`) { triggered = true; interleaved = second.adopt('luna', 'Figa'); }
    return value;
  };
  await first.adopt('miso', 'Rudy');
  await interleaved;
  const saved = (await createLocalStorage(() => memory).load()).adoptions;
  assert.equal(saved.miso.name, 'Rudy');
  assert.equal(saved.luna.name, 'Figa');
});

test('blokada przeglądarki daje błąd bez potwierdzenia adopcji', async () => {
  const memory = memoryStorage();
  memory.setItem = () => { throw new Error('QuotaExceededError'); };
  const events = [];
  const storage = createLocalStorage(() => memory, event => events.push(event));
  await assert.rejects(() => storage.adopt('luna', 'Figa'), /odmówiła zapisu/);
  assert.equal(events.filter(event => event.level === 'success').length, 0);
  await assert.rejects(() => createLocalStorage(() => { throw new Error('SecurityError'); }).load(), /Nie można odczytać/);
});

test('API zapisuje adopcję i losowanie na serwerze, logując potwierdzenie po odpowiedzi', async () => {
  const events = [];
  const record = { name: 'Figa', adoptedAt: '2026-11-16T12:00:00.000Z' };
  const next = { ...createBatch(), adoptions: {} };
  const storage = createApiStorage(async (path, options) => {
    if (path === '/api/cats') return new Response(JSON.stringify({ ...initialBatch, adoptions: {} }));
    assert.equal(options.method, 'POST');
    if (path === '/api/adoptions') {
      assert.deepEqual(JSON.parse(options.body), { catId: 'luna', name: 'Figa' });
      assert.equal(events.some(event => event.text === 'PostgreSQL potwierdził zapis'), false);
      return new Response(JSON.stringify({ adoption: record }), { status: 201 });
    }
    assert.equal(path, '/api/cats/draw');
    assert.deepEqual(JSON.parse(options.body), { batchId: 'initial' });
    return new Response(JSON.stringify(next), { status: 201 });
  }, event => events.push(event));
  assert.equal((await storage.load()).batchId, 'initial');
  assert.deepEqual(await storage.adopt('luna', 'Figa'), record);
  assert.deepEqual(await storage.draw('initial'), next);
  assert.ok(events.some(event => event.text === 'PostgreSQL potwierdził zapis'));
  assert.ok(events.some(event => event.text === 'PostgreSQL: nowa trójka zapisana'));
});

test('awaria API i konflikty nie udają udanego zapisu', async () => {
  const unavailable = createApiStorage(async () => { throw new TypeError('Failed to fetch'); });
  await assert.rejects(() => unavailable.load(), /Nie udało się połączyć/);
  const existing = { name: 'Figa', adoptedAt: '2026-11-16T12:00:00.000Z' };
  const conflict = createApiStorage(async () => new Response(JSON.stringify({ error: 'Konflikt', adoption: existing }), { status: 409 }));
  await assert.rejects(() => conflict.adopt('luna', 'Inna'), error => error instanceof AdoptionConflict && error.adoption.name === 'Figa');
  const stale = createApiStorage(async () => new Response(JSON.stringify({ error: 'Zestaw już się zmienił' }), { status: 409 }));
  await assert.rejects(() => stale.draw('initial'), BatchConflict);
});
