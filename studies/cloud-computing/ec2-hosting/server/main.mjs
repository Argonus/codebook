import { createServer } from 'node:http';
import { fileURLToPath } from 'node:url';
import pg from 'pg';
import { createHandler } from './app.mjs';
import { createRepository } from './repository.mjs';

const pool = new pg.Pool({ max: 4, connectionTimeoutMillis: 5000, query_timeout: 5000 });
pool.on('error', () => console.error('Utracono połączenie z PostgreSQL.'));
const repository = createRepository(pool);
await repository.initialize();
const server = createServer(createHandler(repository, fileURLToPath(new URL('../dist/', import.meta.url))));
server.requestTimeout = 15000;
server.headersTimeout = 10000;
server.listen(Number(process.env.PORT || 8080), '0.0.0.0', () => console.log('Koci azyl EC2: http://localhost:' + (process.env.PORT || 8080)));
for (const signal of ['SIGINT', 'SIGTERM']) {
  process.on(signal, () => {
    server.close(async () => { await pool.end(); process.exit(0); });
    server.closeIdleConnections();
  });
}
