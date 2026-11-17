import { normalizeName, validId } from '../frontend/cats.mjs';
import { securityHeaders, serveStatic } from '../scripts/http.mjs';

function json(response, status, body) {
  response.writeHead(status, { 'Content-Type': 'application/json; charset=utf-8', 'Cache-Control': 'no-store' });
  response.end(JSON.stringify(body));
}

function readBody(request) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    let size = 0;
    request.on('data', chunk => {
      size += chunk.length;
      if (size > 4096) { reject(Object.assign(new Error('Żądanie jest za duże.'), { status: 413 })); return; }
      chunks.push(chunk);
    });
    request.on('end', () => {
      try { resolve(JSON.parse(Buffer.concat(chunks).toString('utf8'))); }
      catch { reject(Object.assign(new Error('Nieprawidłowy JSON.'), { status: 400 })); }
    });
    request.on('error', reject);
  });
}

export function createHandler(repository, directory) {
  return async (request, response) => {
    securityHeaders(response);
    try {
      const { pathname } = new URL(request.url, 'http://localhost');
      if (pathname === '/health') {
        if (request.method !== 'GET') { response.setHeader('Allow', 'GET'); return json(response, 405, { error: 'Dozwolone jest GET.' }); }
        await repository.health();
        return json(response, 200, { status: 'ok', storage: 'postgresql' });
      }
      if (pathname === '/api/cats') {
        if (request.method !== 'GET') { response.setHeader('Allow', 'GET'); return json(response, 405, { error: 'Dozwolone jest GET.' }); }
        return json(response, 200, await repository.state());
      }
      if (pathname === '/api/cats/draw') {
        if (request.method !== 'POST') { response.setHeader('Allow', 'POST'); return json(response, 405, { error: 'Dozwolone jest POST.' }); }
        if (request.headers.origin && ![`http://${request.headers.host}`, `https://${request.headers.host}`].includes(request.headers.origin)) return json(response, 403, { error: 'Żądanie musi pochodzić z tej strony.' });
        if (request.headers['content-type']?.split(';')[0].trim() !== 'application/json') return json(response, 415, { error: 'Wymagany typ application/json.' });
        const body = await readBody(request);
        if (!validId(body?.batchId)) return json(response, 400, { error: 'Nieprawidłowy zestaw kotów.' });
        return json(response, 201, await repository.draw(body.batchId));
      }
      if (pathname === '/api/adoptions') {
        if (request.method === 'GET') return json(response, 200, { adoptions: await repository.all() });
        if (request.method !== 'POST') { response.setHeader('Allow', 'GET, POST'); return json(response, 405, { error: 'Dozwolone są GET i POST.' }); }
        if (request.headers.origin && request.headers.origin !== `http://${request.headers.host}` && request.headers.origin !== `https://${request.headers.host}`) return json(response, 403, { error: 'Żądanie musi pochodzić z tej strony.' });
        if (request.headers['content-type']?.split(';')[0].trim() !== 'application/json') return json(response, 415, { error: 'Wymagany typ application/json.' });
        const body = await readBody(request);
        let name;
        try {
          if (!validId(body?.catId)) throw new Error('Nieprawidłowy identyfikator kota.');
          name = normalizeName(body?.name);
        }
        catch (error) { return json(response, 400, { error: error.message }); }
        const result = await repository.adopt(body.catId, name);
        return json(response, result.created ? 201 : 409, result.created ? { adoption: result.adoption } : { error: 'Ten kot znalazł już dom.', adoption: result.adoption });
      }
      if (pathname.startsWith('/api/')) return json(response, 404, { error: 'Nie znaleziono operacji.' });
      await serveStatic(request, response, directory);
    } catch (error) {
      if (!response.headersSent) json(response, error.status ?? 503, { error: error.status ? error.message : 'Baza jest chwilowo niedostępna. Spróbuj ponownie.' });
      else response.end();
    }
  };
}
