CREATE DATABASE FoodCourt
/*Baza danych i jej instancje zarzadza jest przez instancje SQL SERVER*/
CONTAINMENT=NONE
ON PRIMARY (
 NAME=N'FoodCourt',
 /*Sciezka i nazwa pliku zawierajacego baze danych*/
 FILENAME='C:\Program Files\Microsoft SQL Server\MSSQL15.MSSQLSERVER\MSSQL\DATA\FoodCourt.mdf',
 /*Poczatkowy rozmiar pliku*/
 SIZE=5120KB,
 /*Maksymalny rozmiar pliku*/
 MAXSIZE=UNLIMITED,
 /*Jak przyrasta wielkos pliku bazy anych*/
 FILEGROWTH=1024KB
)
LOG ON (
 NAME=N'FoodCourt_log',
 FILENAME='C:\Program Files\Microsoft SQL Server\MSSQL15.MSSQLSERVER\MSSQL\DATA\FoodCourt.ldf',
 SIZE=1024KB,
 MAXSIZE=2GB,
 FILEGROWTH=1024kb
)
