import { createServer } from 'node:http';
import { fileURLToPath } from 'node:url';
import { securityHeaders, serveStatic } from './http.mjs';

const directory = fileURLToPath(new URL('../dist/', import.meta.url));
const port = Number(process.env.PORT || 4173);
const server = createServer(async (request, response) => {
  securityHeaders(response);
  try { await serveStatic(request, response, directory); }
  catch { response.writeHead(500).end('Błąd podglądu.'); }
});
server.listen(port, '127.0.0.1', () => console.log(`Podgląd S3: http://127.0.0.1:${port}`));
