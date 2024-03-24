USE FoodCourt
GO

SELECT
    CASE
		WHEN CustomerID IS NULL THEN 'Walk In'
		ELSE CC.FirstName + ' ' + CC.LastName 
	END AS Customer
    , COUNT(DISTINCT OO.ID) AS NumberOfOrders
FROM Orders.Orders AS OO
LEFT JOIN Clients.Customers AS CC
	ON OO.CustomerID = CC.ID
GROUP BY OO.CustomerID, CC.FirstName, CC.LastName
ORDER BY NumberOfOrders DESC