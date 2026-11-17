export const cats = [
  { id: 'miso', name: 'Miso', appearance: 'miso' },
  { id: 'luna', name: 'Luna', appearance: 'luna' },
  { id: 'chmurka', name: 'Chmurka', appearance: 'chmurka' },
];
export const initialBatch = { batchId: 'initial', cats };
const names = ['Figa', 'Gucio', 'Pestka', 'Tosia', 'Frodo', 'Puszek', 'Karmel', 'Kropka', 'Tofik', 'Łatka', 'Ptyś', 'Bunia'];

function shuffle(values) {
  const result = [...values];
  for (let i = result.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [result[i], result[j]] = [result[j], result[i]];
  }
  return result;
}

export function createBatch(previousCats = cats) {
  const token = [...crypto.getRandomValues(new Uint32Array(4))].map(value => value.toString(16).padStart(8, '0')).join('');
  const batchId = `batch-${token}`;
  const availableNames = shuffle(names.filter(name => !previousCats.some(cat => cat.name === name)));
  return {
    batchId,
    cats: shuffle(cats).map((cat, index) => ({ id: `${batchId}-${index}`, name: availableNames[index], appearance: cat.appearance })),
  };
}

export function normalizeName(value) {
  if (typeof value !== 'string') throw new Error('Wpisz imię dla kota.');
  const name = value.normalize('NFC').trim().replace(/\s+/gu, ' ');
  if ([...name].length < 2 || [...name].length > 24 || !/^[\p{L}\p{M}\p{N} '’\-]+$/u.test(name)) {
    throw new Error('Imię powinno mieć od 2 do 24 znaków: litery, cyfry, spacje, apostrof lub myślnik.');
  }
  return name;
}

export function validId(value) {
  return typeof value === 'string' && /^[a-z0-9-]{1,80}$/.test(value);
}

export function validateAdoption(catId, value, availableCats = cats) {
  if (!availableCats.some(cat => cat.id === catId)) throw new Error('Nie ma takiego kota w aktualnym zestawie.');
  return normalizeName(value);
}

export function checkBatch(batch) {
  if (!batch || !validId(batch.batchId) || !Array.isArray(batch.cats) || batch.cats.length !== 3 || new Set(batch.cats.map(cat => cat?.id)).size !== 3) throw new Error('Nieprawidłowy zestaw kotów.');
  for (const cat of batch.cats) {
    if (!cat || !validId(cat.id) || normalizeName(cat.name) !== cat.name || !cats.some(known => known.appearance === cat.appearance)) throw new Error('Nieprawidłowy zestaw kotów.');
  }
  return { batchId: batch.batchId, cats: batch.cats };
}
