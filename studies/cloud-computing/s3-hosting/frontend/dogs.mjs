// Identyfikatory i pole cats zachowują zgodność z wcześniejszym zapisem adopcji.
export const dogs = [
  { id: 'miso', name: 'Borys', appearance: 'miso' },
  { id: 'luna', name: 'Luna', appearance: 'luna' },
  { id: 'chmurka', name: 'Tofik', appearance: 'chmurka' },
];
export const initialBatch = { batchId: 'initial', cats: dogs };
const names = ['Figa', 'Gucio', 'Pestka', 'Tosia', 'Frodo', 'Puszek', 'Karmel', 'Kropka', 'Tofik', 'Łatka', 'Ptyś', 'Bunia'];

function shuffle(values) {
  const result = [...values];
  for (let i = result.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [result[i], result[j]] = [result[j], result[i]];
  }
  return result;
}

export function createBatch(previousDogs = dogs) {
  const token = [...crypto.getRandomValues(new Uint32Array(4))].map(value => value.toString(16).padStart(8, '0')).join('');
  const batchId = `batch-${token}`;
  const availableNames = shuffle(names.filter(name => !previousDogs.some(dog => dog.name === name)));
  return {
    batchId,
    cats: shuffle(dogs).map((dog, index) => ({ id: `${batchId}-${index}`, name: availableNames[index], appearance: dog.appearance })),
  };
}

export function normalizeName(value) {
  if (typeof value !== 'string') throw new Error('Wpisz imię dla psa.');
  const name = value.normalize('NFC').trim().replace(/\s+/gu, ' ');
  if ([...name].length < 2 || [...name].length > 24 || !/^[\p{L}\p{M}\p{N} '’\-]+$/u.test(name)) {
    throw new Error('Imię powinno mieć od 2 do 24 znaków: litery, cyfry, spacje, apostrof lub myślnik.');
  }
  return name;
}

export function validId(value) {
  return typeof value === 'string' && /^[a-z0-9-]{1,80}$/.test(value);
}

export function validateAdoption(dogId, value, availableDogs = dogs) {
  if (!availableDogs.some(dog => dog.id === dogId)) throw new Error('Nie ma takiego psa w aktualnym zestawie.');
  return normalizeName(value);
}

export function checkBatch(batch) {
  if (!batch || !validId(batch.batchId) || !Array.isArray(batch.cats) || batch.cats.length !== 3 || new Set(batch.cats.map(dog => dog?.id)).size !== 3) throw new Error('Nieprawidłowy zestaw psów.');
  for (const dog of batch.cats) {
    if (!dog || !validId(dog.id) || normalizeName(dog.name) !== dog.name || !dogs.some(known => known.appearance === dog.appearance)) throw new Error('Nieprawidłowy zestaw psów.');
  }
  return { batchId: batch.batchId, cats: batch.cats };
}
