import { cp, mkdir, readFile, rm, writeFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';

const root = fileURLToPath(new URL('../', import.meta.url));
const { hostingMode: mode } = JSON.parse(await readFile(`${root}package.json`, 'utf8'));
if (!['s3', 'ec2'].includes(mode)) throw new Error('Nieprawidłowy hostingMode w package.json.');
const author = JSON.parse(await readFile(`${root}author.json`, 'utf8'));
for (const field of ['firstName', 'lastName', 'group', 'studentId']) {
  if (typeof author[field] !== 'string' || !author[field].trim()) throw new Error(`Brak danych autora: ${field}`);
}
{
  const destination = `${root}dist`;
  await rm(destination, { recursive: true, force: true });
  await mkdir(destination, { recursive: true });
  await cp(`${root}frontend`, destination, { recursive: true });
  await writeFile(`${destination}/config.mjs`, `export const config = ${JSON.stringify({ mode, author }, null, 2)};\n`);
  console.log(`Gotowe: dist (${author.firstName} ${author.lastName} · ${mode.toUpperCase()})`);
}
