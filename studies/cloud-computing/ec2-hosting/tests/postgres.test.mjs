import assert from 'node:assert/strict';
import { randomUUID } from 'node:crypto';
import { createServer } from 'node:http';
import { fileURLToPath } from 'node:url';
import { test } from 'node:test';
import pg from 'pg';
import { createRepository } from '../server/repository.mjs';
import { createHandler } from '../server/app.mjs';

test('prawdziwy PostgreSQL: adopcja, współbieżność i odczyt po ponownym połączeniu', async () => {
  const schema = `test_${randomUUID().replaceAll('-', '')}`;
  const admin = new pg.Pool({ max: 1, connectionTimeoutMillis: 5000 });
  const pool = new pg.Pool({ max: 4, connectionTimeoutMillis: 5000, options: `-c search_path=${schema}` });
  let server;
  let restarted;
  try {
    await admin.query(`CREATE SCHEMA ${schema}`);
    const repository = createRepository(pool);
    await repository.initialize();
    server = createServer(createHandler(repository, fileURLToPath(new URL('../dist/', import.meta.url))));
    await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
    const base = `http://127.0.0.1:${server.address().port}`;
    const responses = await Promise.all(['Figa', 'Łapka'].map(name => fetch(`${base}/api/adoptions`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ catId: 'luna', name }) })));
    assert.deepEqual(responses.map(response => response.status).sort(), [201, 409]);
    const result = await (await fetch(`${base}/api/adoptions`)).json();
    assert.equal(Object.keys(result.adoptions).length, 1);
    const name = result.adoptions.luna.name;
    assert.ok(['Figa', 'Łapka'].includes(name));
    await pool.end();
    restarted = new pg.Pool({ max: 1, options: `-c search_path=${schema}` });
    const reconnected = createRepository(restarted);
    await reconnected.initialize();
    assert.equal((await reconnected.all()).luna.name, name);
    assert.equal((await admin.query(`SELECT count(*) FROM ${schema}.adoptions`)).rows[0].count, '1');
  } finally {
    if (server) await new Promise(resolve => { server.close(resolve); server.closeAllConnections(); });
    if (!pool.ended) await pool.end();
    if (restarted) await restarted.end();
    await admin.query(`DROP SCHEMA IF EXISTS ${schema} CASCADE`);
    await admin.end();
  }
});
