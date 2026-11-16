# Kocia adopcja — EC2

Piotr Rybarczyk · grupa MZ101 · nr albumu 10117

Samodzielny projekt: prosty landing z trzema animowanymi kotami, nadawaniem imion i widocznym logiem operacji. Nie wymaga plików z projektu S3 ani z katalogu nadrzędnego.

## Uruchomienie lokalne

Wymagany Docker Engine lub Docker Desktop z Docker Compose.

```sh
cp .env.example .env
```

W `.env` ustaw własne długie hasło `POSTGRES_PASSWORD`, następnie:

```sh
docker compose up -d --build --wait
```

Strona: <http://127.0.0.1:8080>. Gotowość aplikacji i bazy: <http://127.0.0.1:8080/health>.

## Jak działa zapis

Przeglądarka → `POST /api/adoptions` → Node.js → PostgreSQL.

Po zatwierdzeniu imienia API zapisuje identyfikator kota, imię i czas adopcji. Dopiero po odpowiedzi potwierdzającej zapis otwierają się drzwiczki i kot wychodzi na poduszkę. Odświeżenie wywołuje `GET /api/adoptions` i odtwarza stan z bazy. Adopcje są wspólne dla wszystkich odwiedzających. Ponowna próba adopcji tego samego kota zwraca HTTP 409 i nie zmienia pierwszego zapisu.

Log na stronie pokazuje rzeczywiste żądania HTTP, ich status i potwierdzenie zapisu. Błąd bazy nie powoduje pozornej adopcji ani przełączenia na localStorage.

PostgreSQL ma nazwany wolumen `postgres-data`, zachowywany po restarcie i odtworzeniu kontenera. Port bazy nie jest publikowany na hoście. Polecenie `docker compose down` zachowuje dane; opcja `-v` usuwa wolumen wraz z adopcjami. Wolumen na EC2 nie zastępuje kopii zapasowej i może zostać utracony po usunięciu dysku instancji.

## Sprawdzenie

```sh
docker compose exec -T web npm test
docker compose exec -T web npm run test:integration
```

Test integracyjny tworzy i usuwa własny schemat testowy; nie zmienia demonstracyjnych adopcji. Sprawdza również równoczesną adopcję i zachowanie stanu po ponownym połączeniu.

Ręcznie: nadaj imię, kliknij „Adoptuj”, sprawdź wyjście z klatki, odśwież stronę i otwórz ją w drugiej przeglądarce. EC2 pokaże ten sam stan. Można też wykonać `docker compose restart`, by sprawdzić trwałość wolumenu.

## Przygotowanie do AWS EC2

1. Na prywatnym koncie AWS przygotuj małą instancję Linux z co najmniej 1 GiB RAM, dyskiem EBS i publicznym IPv4. Zainstaluj Docker Engine oraz plugin Compose według [instrukcji Docker dla Ubuntu](https://docs.docker.com/engine/install/ubuntu/).
2. Przenieś i rozpakuj archiwum tego projektu na instancji. Utwórz `.env` z własnym hasłem. Ustaw `APP_BIND_ADDRESS=0.0.0.0` i `APP_PORT=80`.
3. W Security Group udostępnij HTTP/TCP 80 odwiedzającym. Jeśli używasz SSH, ogranicz TCP 22 do własnego adresu. Baza PostgreSQL nie wymaga publicznej reguły.
4. Uruchom `docker compose up -d --build --wait`. Sprawdź `http://PUBLICZNY_ADRES_IP/` z innego urządzenia.
5. Do osobnego raportu dodaj publiczny link, zrzut strony z kotem po adopcji i logiem, konfigurację instancji oraz opis powyższego przepływu danych.

Instancja ma działać 20.11.2026 od 20:00 do 22:00, zgodnie z treścią zadania; zatrzymaj ją po akceptacji prowadzącego. Zwykły publiczny IPv4 może zmienić się po ponownym uruchomieniu, więc sprawdź link przed oddaniem. Dobierz zasoby i sprawdź opłaty na własnym koncie; projekt nie zakłada darmowego Free Tier.

## Kod i archiwum

Do budowania i pakowania lokalnie potrzebny jest Node.js 24 lub nowszy. Do samego `npm run package` nie potrzeba instalować zależności.

```sh
npm run package
```

Wynik: `.artifacts/ec2-hosting.tar.gz`. Archiwum zawiera kod, instrukcję i gotowy `dist/`; pomija `.env`, `node_modules` i wcześniejsze archiwa. Po rozpakowaniu Docker pobierze zależności z publicznego npm. Dane autora są w `author.json`; po ich zmianie przebuduj obraz.

`frontend/` zawiera stronę i ilustracje SVG, `server/` — API i bazę, `scripts/` — budowanie i pakowanie, `tests/` — testy. Brak usług analitycznych, zewnętrznych fontów i kluczy AWS w kodzie.

## Status

Implementacja jest przygotowana i sprawdzona lokalnie. Zasoby AWS nie zostały utworzone, a publiczny URL i końcowy raport ze zrzutem wdrożenia powstaną po publikacji. Automatyczna weryfikacja wyglądu w przeglądarce była niedostępna; wykonane testy nie zastępują kontroli wizualnej.

Źródła: [node-postgres: zapytania parametryzowane](https://node-postgres.com/features/queries), [Docker: trwałe wolumeny](https://docs.docker.com/engine/storage/volumes/), [AWS: zatrzymanie i uruchomienie EC2](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/Stop_Start.html).
