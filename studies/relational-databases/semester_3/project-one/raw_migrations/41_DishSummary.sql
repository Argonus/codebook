-- Prepare a table with dish sales data
USE FoodCourt
GO

WITH DishSales AS (
    SELECT 
        RD.ID AS DishID
        , RD.Name AS DishName
        , CAST(OO.RequestedDateTime AS DATE) AS SaleDate
        , OOI.Quantity
    FROM Orders.Orders AS OO
    INNER JOIN Orders.OrderItems AS OOI 
		ON OOI.OrderID = OO.ID
    INNER JOIN Resources.Dishes AS RD
		ON RD.ID = OOI.DishID
	WHERE OO.Status IN ('Ready', 'Completed')
)

-- Pivot the sales data by date
SELECT DishName,
       SUM([2023-05-01]) AS '2023-05-01', 
       SUM([2023-05-02]) AS '2023-05-02', 
       SUM([2023-05-03]) AS '2023-05-03', 
       SUM([2023-05-04]) AS '2023-05-04', 
       SUM([2023-05-05]) AS '2023-05-05', 
       SUM([2023-05-06]) AS '2023-05-06', 
       SUM([2023-05-07]) AS '2023-05-07'
FROM DishSales
PIVOT (
    SUM(Quantity)
    FOR SaleDate IN ([2023-05-01], [2023-05-02], [2023-05-03], [2023-05-04], [2023-05-05], [2023-05-06], [2023-05-07])
) AS DishPivot
GROUP BY DishName