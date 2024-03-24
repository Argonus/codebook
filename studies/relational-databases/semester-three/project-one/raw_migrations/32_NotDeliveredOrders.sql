USE FoodCourt
GO

SELECT
	RL.ID AS LocationId
	, OO.ID AS OrderID
	, OO.RequestedDateTime AS RequestedDateTime
	, CC.PhoneNumber AS CustomerPhoneNumber
	, CC.Email AS CustomerEmail
FROM
	Restaurants.Locations AS RL
INNER JOIN Orders.Orders AS OO
	ON RL.ID = OO.LocationID
INNER JOIN Clients.Customers AS CC
	ON OO.CustomerID = CC.ID
INNER JOIN Orders.Deliveries AS OD
	ON OD.OrderID = OO.ID
WHERE
	OO.Status NOT IN ('Ready', 'Completed')
	AND OO.OrderType = 'Delivery'
	AND OD.Status NOT IN ('Completed')
	AND CAST(OO.RequestedDateTime AS DATE) < GETDATE();
