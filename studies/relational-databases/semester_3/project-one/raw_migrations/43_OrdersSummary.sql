USE FoodCourt

SELECT
	OO.OrderType
	, COUNT(DISTINCT(OO.ID)) AS OrderCount
	, SUM(OI.Quantity * RD.Price) AS TotalPrice
FROM Orders.Orders AS OO
INNER JOIN Orders.OrderItems AS OI
	ON OO.ID = OI.OrderID
INNER JOIN Orders.OrderTypes AS OT
	ON OO.OrderType = OT.Type
INNER JOIN Resources.Dishes AS RD
	ON RD.ID = OI.DishID
GROUP BY OO.OrderType
ORDER BY SUM(OI.Quantity * RD.Price) 