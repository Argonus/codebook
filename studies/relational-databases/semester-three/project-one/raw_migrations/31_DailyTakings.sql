USE FoodCourt
GO


SELECT
	RL.ID AS LocationId
	, OO.OrderType AS OrderType
	, CAST(OO.RequestedDateTime AS DATE) AS Date
	, SUM(OOI.Quantity * RD.Price) AS DailyTakings
FROM
	Restaurants.Locations AS RL
LEFT JOIN
	Orders.Orders AS OO
	ON OO.LocationID = RL.ID AND OO.Status IN ('Ready', 'Completed')
LEFT JOIN
	Orders.OrderItems AS OOI
	ON OOI.OrderID = OO.ID
LEFT JOIN
	Resources.Dishes AS RD
	ON OOI.DishID = RD.ID
GROUP BY RL.ID, OO.OrderType, CAST(OO.RequestedDateTime AS DATE)
ORDER BY DATE, LocationId, DailyTakings DESC


