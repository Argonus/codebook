USE FoodCourt
GO

WITH CustomerOrders AS (
	SELECT OO.CustomerID FROM Orders.Orders AS OO WHERE OO.OrderType = 'Delivery'
	INTERSECT
	SELECT OO.CustomerID FROM Orders.Orders AS OO WHERE OO.OrderType = 'DineIn'
)
SELECT 
	CC.ID
	, CC.FirstName + ' ' + CC.LastName AS CustomerName
	, COALESCE(CC.Email, CC.PhoneNumber, 'N/A') ContactDetails
	, CC.Blocked
FROM
	CustomerOrders
INNER JOIN Clients.Customers AS CC
	ON CC.ID = CustomerOrders.CustomerID

