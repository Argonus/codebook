USE FoodCourt
GO

SELECT
	RT.ID AS TableID
	, RL.Name AS LocationName
	, COUNT(R.ID) AS ReservationsCount
	, AVG(R.Seats) AS AvgSeatsNumber
	, AVG(DATEDIFF(MINUTE, R.ReservationHour, DATEADD(SECOND, R.ReservationTime, R.ReservationHour))) AS AvgTimeInMinutes
FROM Restaurants.Tables AS RT
INNER JOIN Restaurants.Locations AS RL
	ON RL.ID = RT.LocationID
INNER JOIN Restaurants.TableReservations AS TR
    ON TR.TableID = RT.ID
INNER JOIN Restaurants.Reservations AS R
    ON R.ID = TR.ReservationID
GROUP BY RT.ID, RL.Name
ORDER BY ReservationsCount DESC