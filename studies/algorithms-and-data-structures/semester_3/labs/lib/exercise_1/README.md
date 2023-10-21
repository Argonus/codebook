## Implementation
Dany jest zasobnik, w którym są kule białe i czarne, przyczym liczba kul czarnych jest nieparzysta. Wielokrotnie, aż dosytuacji, 
w której w zasobniku pozostanie jedna kula, następuje losowanie dwóch kul. Jeżeli kule są różnego koloru to do zasobnika 
wraca kula czarna, w przeciwnym przypadku obie kule są odrzucane. Wynikiem działania algorytmu jest kolor ostatniej kuli 
pozostającej w zasobniku.
## Invariant Rule
Liczba kul czarnych przez cały czas trwania algorytmu jest nieparzysta.

## Result & Observation
Na koniec dzialania algorytmu pozostaje kula czarna
Wynika to, że liczbę kul czarnych zmniejszamy tylko o 0 lub 2, natomiast liczbę kul białych zmniejszamy
1 lub 2.