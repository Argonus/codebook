USE FoodCourt
GO

WITH DishesPopularity AS (
	SELECT
		CAST(OO.RequestedDateTime AS DATE) AS DishDate
		, RD.Name AS DishName
		, SUM(OOI.Quantity) As DishQuantity
		, ROW_NUMBER() OVER (PARTITION BY CAST(OO.RequestedDateTime AS DATE) ORDER BY SUM(OOI.Quantity) DESC) AS Ranking
	FROM
		Resources.Dishes AS RD
	INNER JOIN
		Orders.OrderItems AS OOI
		ON OOI.DishID = RD.ID
	INNER JOIN
		Orders.Orders AS OO
		ON OO.ID = OOI.OrderID
	GROUP BY
		CAST(OO.RequestedDateTime AS DATE), RD.Name
) SELECT DishDate, DishName, DishQuantity FROM DishesPopularity WHERE Ranking = 1;