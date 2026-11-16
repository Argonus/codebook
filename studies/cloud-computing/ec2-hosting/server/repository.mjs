import { readFile } from 'node:fs/promises';
import { cats } from '../frontend/cats.mjs';

function adoption(row) {
  return { name: row.name, adoptedAt: row.adopted_at.toISOString() };
}

export function createRepository(pool) {
  return {
    async initialize() {
      await pool.query(await readFile(new URL('./schema.sql', import.meta.url), 'utf8'));
      await pool.query('INSERT INTO cats (id) SELECT unnest($1::text[]) ON CONFLICT DO NOTHING', [cats.map(cat => cat.id)]);
    },
    async health() { await pool.query('SELECT 1'); },
    async all() {
      const { rows } = await pool.query('SELECT cat_id, name, adopted_at FROM adoptions ORDER BY cat_id');
      return Object.fromEntries(rows.map(row => [row.cat_id, adoption(row)]));
    },
    async adopt(catId, name) {
      const { rows } = await pool.query('INSERT INTO adoptions (cat_id, name) VALUES ($1, $2) ON CONFLICT (cat_id) DO NOTHING RETURNING name, adopted_at', [catId, name]);
      if (rows.length) return { created: true, adoption: adoption(rows[0]) };
      const existing = await pool.query('SELECT name, adopted_at FROM adoptions WHERE cat_id = $1', [catId]);
      return { created: false, adoption: adoption(existing.rows[0]) };
    },
  };
}
