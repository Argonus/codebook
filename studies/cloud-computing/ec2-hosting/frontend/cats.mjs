export const cats = [
  { id: 'miso', name: 'Miso' },
  { id: 'luna', name: 'Luna' },
  { id: 'chmurka', name: 'Chmurka' },
];

export function normalizeName(value) {
  if (typeof value !== 'string') throw new Error('Wpisz imię dla kota.');
  const name = value.normalize('NFC').trim().replace(/\s+/gu, ' ');
  const length = [...name].length;
  if (length < 2 || length > 24 || !/^[\p{L}\p{M}\p{N} '’\-]+$/u.test(name)) {
    throw new Error('Imię powinno mieć od 2 do 24 znaków: litery, cyfry, spacje, apostrof lub myślnik.');
  }
  return name;
}

export function validateAdoption(catId, value) {
  if (!cats.some(cat => cat.id === catId)) throw new Error('Nie ma takiego kota w naszym azylu.');
  return normalizeName(value);
}
