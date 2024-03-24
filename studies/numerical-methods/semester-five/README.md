# Rodzaje bledow
## Opisu Problemu
--- model fizyczny jest przyblizeniem zjawiska rzeczywistego.
--- sformuulowane zadanie obliczeniowe jest zbyt skomplikowane.

## Bledy obciecia
--- dzialaialania sa nieskonczone.
--- dzialania na wielkosciach nieskonczenie malych

## Bledy poczatkowe
--- dane z pomiarow
--- zakraglenie licz stalych lub niewimietnych

## Bledy zaokraglenia
--- niedokladna reprezentacja liczby rzeczywistej 
### Blad bezwzgledny
--- |x - x0|
### Blad wzgledny
--- |x - x0|/|x| * 100%

# Kryteria doboru algorytmow
- Jakosc numerycza wynikow
- Zlozonosc obliczeniowa
- Czas obliczen
## Konwergencja
-- zdolnosc algorytmu do zbiegania do rozwiazania
## Dywergencja
-- zdolnosc algorytmu do rozbiegania od rozwiazania

# Zapis zmienno przecinkowy
## Mantysa i cecha (wykladnik)
256.78 = 2.5678 * 10^2
- Mantysa = 2568
- Cecha = 2
---- Float
- Mantysa = 23 bity
- Cecha = 8 bitow
- Znak = 1 bit
---- Double
- Mantysa = 52 bity
- Cecha = 11 bitow
- Znak = 1 bit

# Uklady rownan liniowych
## jednorodne i niejednorodne
## Metody rozwiazywania

### Metody skonczone
- bledy zaokraglenia

#### Eliminacja Gaussa
##### Wady
- blad dzieelenia przez zero
- narastanie bledu zaokraglenia
##### Zalety
- mniejsza liczba operacji niz przy metodzie Cramera
##### Przyklad
10x1 - 7x2 + 0x3 = 6
0x1 - 0.1x2 + 6x3 = 5.8
0x1 + 2.5x2 + 5x3 = 0
# Macierz
|10 -7 0|  |x1| |6|
|0 -0.1 6| |x2| |5.8|
|0 2.5 5|  |x3| |0|
# Zero na przekatnej
|10 -7 0|  |x1| |6|
|0 -0.1 6| |x2| |5.8| * 25
|0 0 155|  |x3| |145|
#### Eliminacja LU
Rozklad macierzy A na iloczyn dwoch macierzy L i U
- L - macierz trojkatna dolna
- U - macierz trojkatna gorna
#### Rozklad Choleskiego
#### Rozklad Crouta
#### Rozklad Doolittle

### Metody iteracyjne
- blad metody i bledy zaokraglenia