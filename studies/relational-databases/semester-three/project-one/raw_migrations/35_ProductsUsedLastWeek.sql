USE FoodCourt
Go

DECLARE @CurrentWeekday INT = DATEPART(WEEKDAY, GETDATE())
DECLARE @DateDiff INT = -(7 * @CurrentWeekday % 7);
DECLARE @PrevMonday DATE = DATEADD(DAY, @DateDiff - 8, GETDATE())

SELECT
	RP.Name
	, COUNT(DISTINCT(OO.ID)) AS OrdersCount
	, SUM(OOI.Quantity) AS ProductsUsed
FROM Restaurants.Locations AS RL
INNER JOIN Orders.Orders AS OO
	ON OO.LocationID = RL.ID
INNER JOIN Orders.OrderItems AS OOI
	ON OOI.OrderID = OO.ID
INNER JOIN Resources.Dishes AS RD
	ON RD.ID = OOI.DishID
INNER JOIN Resources.DishProducts AS RDP
	ON RDP.DishID = RD.ID
INNER JOIN Resources.Products AS RP
	ON RP.ID = RDP.ProductID
WHERE 
	RL.ID = 137
	AND OO.RequestedDateTime BETWEEN @PrevMonday AND DATEADD(DAY, 7, @PrevMonday)
	AND OO.Status NOT IN ('Pending', 'Accepted')
GROUP BY RP.Name
ORDER BY RP.Name
