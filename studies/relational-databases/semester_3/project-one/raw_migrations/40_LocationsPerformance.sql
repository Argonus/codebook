USE FoodCourt 
GO


SELECT 
	RLD.LocationName
	, SUM(OOI.Quantity * RD.Price) AS TotalSales
	, NTILE(3) OVER (ORDER BY SUM(OOI.Quantity * RD.Price) DESC) AS PerformanceTier
FROM Orders.Orders AS OO
INNER JOIN Orders.OrderItems AS OOI
	ON OOI.OrderID = OO.ID
INNER JOIN Resources.Dishes AS RD
	ON RD.ID = OOI.DishID
INNER JOIN Restaurants.LocationDetails AS RLD
    ON OO.LocationID = RLD.LocationId
GROUP BY RLD.LocationName