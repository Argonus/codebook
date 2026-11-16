import { execFileSync } from 'node:child_process';
import { mkdir } from 'node:fs/promises';
import { basename, dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = fileURLToPath(new URL('../', import.meta.url)).replace(/\/$/, '');
const destination = join(root, '.artifacts');
await mkdir(destination, { recursive: true });
const archive = join(destination, `${basename(root)}.tar.gz`);
execFileSync('tar', ['-czf', archive, '--exclude=node_modules', '--exclude=.env', '--exclude=.artifacts', '--exclude=.DS_Store', '-C', dirname(root), basename(root)]);
console.log(archive);
