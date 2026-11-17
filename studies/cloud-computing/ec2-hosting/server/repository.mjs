import { readFile } from 'node:fs/promises';
import { cats, createBatch } from '../frontend/cats.mjs';

function adoption(row) { return { name: row.adopted_name, adoptedAt: row.adopted_at.toISOString() }; }
function conflict(message) { return Object.assign(new Error(message), { status: 409 }); }

async function readState(connection) {
  const { rows } = await connection.query('SELECT s.batch_id, c.id, c.name, c.appearance, a.name AS adopted_name, a.adopted_at FROM shelter s JOIN cats c ON c.batch_id = s.batch_id LEFT JOIN adoptions a ON a.cat_id = c.id WHERE s.id = true ORDER BY c.position');
  if (rows.length !== 3) throw new Error('Nieprawidłowa liczba kotów w aktywnym zestawie.');
  return {
    batchId: rows[0].batch_id,
    cats: rows.map(row => ({ id: row.id, name: row.name, appearance: row.appearance })),
    adoptions: Object.fromEntries(rows.filter(row => row.adopted_at).map(row => [row.id, adoption(row)])),
  };
}

export function createRepository(pool) {
  async function lockedBatch(operation) {
    const connection = await pool.connect();
    try {
      await connection.query('BEGIN');
      const { rows } = await connection.query('SELECT batch_id FROM shelter WHERE id = true FOR UPDATE');
      const result = await operation(connection, rows[0].batch_id);
      await connection.query('COMMIT');
      return result;
    } catch (error) {
      await connection.query('ROLLBACK');
      throw error;
    } finally { connection.release(); }
  }
  return {
    async initialize() {
      await pool.query(await readFile(new URL('./schema.sql', import.meta.url), 'utf8'));
      for (const [position, cat] of cats.entries()) {
        await pool.query('INSERT INTO cats (id, batch_id, name, appearance, position) VALUES ($1, $2, $3, $4, $5) ON CONFLICT (id) DO UPDATE SET name = COALESCE(cats.name, EXCLUDED.name), appearance = COALESCE(cats.appearance, EXCLUDED.appearance), position = COALESCE(cats.position, EXCLUDED.position)', [cat.id, 'initial', cat.name, cat.appearance, position]);
      }
      await pool.query("INSERT INTO shelter (id, batch_id) VALUES (true, 'initial') ON CONFLICT DO NOTHING");
    },
    async health() { await pool.query('SELECT 1'); },
    async state() { return readState(pool); },
    async all() { return (await readState(pool)).adoptions; },
    async adopt(catId, name) {
      return lockedBatch(async (connection, batchId) => {
        const cat = await connection.query('SELECT batch_id FROM cats WHERE id = $1', [catId]);
        if (!cat.rows.length) throw Object.assign(new Error('Nie ma takiego kota w naszym azylu.'), { status: 400 });
        if (cat.rows[0].batch_id !== batchId) throw conflict('Zestaw kotów już się zmienił. Odświeżamy stronę.');
        const { rows } = await connection.query('INSERT INTO adoptions (cat_id, name) VALUES ($1, $2) ON CONFLICT (cat_id) DO NOTHING RETURNING name AS adopted_name, adopted_at', [catId, name]);
        if (rows.length) return { created: true, adoption: adoption(rows[0]) };
        const existing = await connection.query('SELECT name AS adopted_name, adopted_at FROM adoptions WHERE cat_id = $1', [catId]);
        return { created: false, adoption: adoption(existing.rows[0]) };
      });
    },
    async draw(expectedBatchId) {
      return lockedBatch(async (connection, batchId) => {
        if (expectedBatchId !== batchId) throw conflict('Zestaw kotów już się zmienił. Odświeżamy stronę.');
        const current = await readState(connection);
        if (!current.cats.every(cat => current.adoptions[cat.id])) throw conflict('Najpierw adoptuj wszystkie trzy koty.');
        const next = createBatch(current.cats);
        for (const [position, cat] of next.cats.entries()) {
          await connection.query('INSERT INTO cats (id, batch_id, name, appearance, position) VALUES ($1, $2, $3, $4, $5)', [cat.id, next.batchId, cat.name, cat.appearance, position]);
        }
        await connection.query('UPDATE shelter SET batch_id = $1 WHERE id = true', [next.batchId]);
        return { ...next, adoptions: {} };
      });
    },
  };
}
