import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { fileURLToPath } from 'node:url';
import { test } from 'node:test';
import { createHandler } from '../server/app.mjs';

const directory = fileURLToPath(new URL('../frontend/', import.meta.url));

async function withServer(t, repository) {
  const server = createServer(createHandler(repository, directory));
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  t.after(() => new Promise(resolve => { server.close(resolve); server.closeAllConnections(); }));
  return `http://127.0.0.1:${server.address().port}`;
}

test('API waliduje żądania, zapisuje i obsługuje konflikt bez nadpisania', async t => {
  const records = {};
  const base = await withServer(t, {
    async health() {}, async all() { return records; },
    async adopt(id, name) {
      if (records[id]) return { created: false, adoption: records[id] };
      records[id] = { name, adoptedAt: new Date().toISOString() };
      return { created: true, adoption: records[id] };
    },
  });
  const adopt = body => fetch(`${base}/api/adoptions`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  assert.equal((await fetch(`${base}/health`)).status, 200);
  assert.equal((await adopt({ catId: 'missing', name: 'Figa' })).status, 400);
  assert.equal((await adopt({ catId: 'luna', name: '<script>' })).status, 400);
  assert.equal((await adopt({ catId: 'luna', name: 'Łapka' })).status, 201);
  const conflict = await adopt({ catId: 'luna', name: 'Inna' });
  assert.equal(conflict.status, 409);
  assert.equal((await conflict.json()).adoption.name, 'Łapka');
  assert.equal((await (await fetch(`${base}/api/adoptions`)).json()).adoptions.luna.name, 'Łapka');
  assert.equal((await fetch(`${base}/api/adoptions`, { method: 'DELETE' })).status, 405);
  assert.equal((await fetch(`${base}/api/unknown`)).status, 404);
  assert.equal((await fetch(`${base}/api/adoptions`, { method: 'POST', body: 'text' })).status, 415);
  assert.equal((await fetch(`${base}/api/adoptions`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{broken' })).status, 400);
  assert.equal((await adopt({ catId: 'miso', name: 'x'.repeat(5000) })).status, 413);
  assert.equal((await fetch(`${base}/api/adoptions`, { method: 'POST', headers: { 'Content-Type': 'application/json', Origin: 'https://foreign.example' }, body: '{}' })).status, 403);
  const page = await fetch(base);
  assert.equal(page.status, 200);
  assert.match(page.headers.get('content-security-policy'), /default-src 'self'/);
  assert.equal((await fetch(`${base}/.env`)).status, 404);
  assert.equal((await fetch(`${base}/%2e%2e%2fpackage.json`)).status, 404);
});

test('awaria bazy daje 503 bez udawania zapisu', async t => {
  const unavailable = async () => { throw new Error('private database details'); };
  const base = await withServer(t, { health: unavailable, all: unavailable, adopt: unavailable });
  for (const path of ['/health', '/api/adoptions']) {
    const response = await fetch(`${base}${path}`);
    assert.equal(response.status, 503);
    assert.doesNotMatch(await response.text(), /private database details/);
  }
  const response = await fetch(`${base}/api/adoptions`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ catId: 'miso', name: 'Rudy' }) });
  assert.equal(response.status, 503);
});
