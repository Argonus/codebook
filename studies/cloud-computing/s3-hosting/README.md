# Kocia adopcja — S3

Piotr Rybarczyk · grupa MZ101 · nr albumu 10117

Samodzielny projekt: prosty landing z trzema animowanymi kotami, nadawaniem imion i widocznym logiem operacji. Nie wymaga backendu, bazy, plików projektu EC2 ani instalowania zależności npm.

## Uruchomienie lokalne

Wymagany Node.js 24 lub nowszy.

```sh
npm run dev
```

Strona: <http://127.0.0.1:4173>. Podgląd HTTP jest potrzebny, ponieważ nie należy polegać na localStorage przy otwieraniu pliku bezpośrednio z dysku.

## Jak działa zapis

S3 dostarcza statyczne pliki HTML, CSS, JavaScript i SVG. Adopcja wykonuje się w przeglądarce:

przeglądarka → `localStorage.setItem('koci-azyl.adoptions.v1.ID_KOTA', ...)`.

Zapis obejmuje identyfikator kota, nadane imię i czas adopcji. Dopiero po udanym zapisie otwierają się drzwiczki, a kot wychodzi na poduszkę. Odświeżenie strony odczytuje localStorage i odtwarza stan. Log pokazuje rzeczywiste operacje `getItem` i `setItem`; adopcja nie wysyła żądań do API.

Każdy odwiedzający ma własny stan. Inna przeglądarka lub urządzenie pokaże niezależny azyl. Dane są związane z adresem strony (origin): lokalny podgląd i publiczny adres S3 mają osobne zapisy. Usunięcie danych witryny usuwa adopcje. W trybie prywatnym dane mogą zniknąć po zamknięciu sesji. Zablokowany albo uszkodzony zapis powoduje komunikat błędu; kot nie wychodzi po nieudanym zapisie.

## Sprawdzenie

```sh
npm test
```

Ręcznie: nadaj imię, kliknij „Adoptuj”, sprawdź log, odśwież stronę i potwierdź obecność kota poza klatką. W innej przeglądarce ten kot powinien nadal czekać na adopcję. Dwie karty tej samej strony otrzymują aktualizacje przez zdarzenie `storage`.

## Przygotowanie do AWS S3

```sh
npm run build
```

Do hostingu służy wyłącznie zawartość `dist/`, z `index.html` w katalogu głównym bucketa.

1. Na prywatnym koncie AWS utwórz osobny bucket przeznaczony na tę stronę.
2. W Properties włącz Static website hosting i ustaw dokument indeksowy `index.html`.
3. Skonfiguruj odczyt publiczny plików tej witryny (`s3:GetObject` dla `arn:aws:s3:::NAZWA_BUCKETA/*`) i ustawienia Block Public Access odpowiednie do wybranego hostingu. Nie udostępniaj publicznego zapisu.
4. Wgraj zawartość `dist/`. Pliki `.mjs` muszą mieć Content-Type `text/javascript`; HTML — `text/html`, CSS — `text/css`, SVG — `image/svg+xml`.
5. Otwórz podany przez S3 Website endpoint, sprawdź adopcję i odświeżenie. Do osobnego raportu dodaj link, zrzut strony z logiem oraz opis konfiguracji i localStorage.

Przy użyciu AWS CLI można wgrać pliki dwoma poleceniami, aby jawnie ustawić typ modułów JavaScript. Zastąp profil i nazwę bucketa własnymi danymi. Te polecenia wykonują publikację i nie są częścią lokalnego budowania:

```sh
aws --profile PRYWATNY_PROFIL s3 sync dist/ s3://NAZWA_BUCKETA/ --exclude '*.mjs' --cache-control no-cache
aws --profile PRYWATNY_PROFIL s3 cp dist/ s3://NAZWA_BUCKETA/ --recursive --exclude '*' --include '*.mjs' --content-type text/javascript --cache-control no-cache
```

Bezpośredni S3 Website endpoint działa przez HTTP. HTTPS wymaga dodatkowej usługi, np. CloudFront; nie jest elementem tej implementacji. Termin oddania: 21.11.2026, 23:00.

## Kod i archiwum

```sh
npm run package
```

Wynik: `.artifacts/s3-hosting.tar.gz`. Archiwum zawiera kod, instrukcję i gotowy `dist/`; pomija `.env`, `node_modules` i wcześniejsze archiwa. Rozpakowany projekt działa samodzielnie. Dane autora są w `author.json`; po ich zmianie wykonaj ponownie `npm run build`.

`frontend/` zawiera stronę i ilustracje SVG, `scripts/` — budowanie, podgląd i pakowanie, `tests/` — testy. Brak usług analitycznych, zewnętrznych fontów i kluczy AWS w kodzie.

## Status

Implementacja jest przygotowana i sprawdzona lokalnie. Bucket nie został utworzony, a publiczny URL i końcowy raport ze zrzutem wdrożenia powstaną po publikacji. Automatyczna weryfikacja wyglądu w przeglądarce była niedostępna; wykonane testy nie zastępują kontroli wizualnej.

Źródła: [AWS: hosting statyczny S3](https://docs.aws.amazon.com/AmazonS3/latest/userguide/WebsiteHosting.html), [AWS: Website endpoints](https://docs.aws.amazon.com/AmazonS3/latest/userguide/WebsiteEndpoints.html), [MDN: localStorage](https://developer.mozilla.org/en-US/docs/Web/API/Window/localStorage).
