SELECT
	OO.ID AS OrderId
	, OO.DeletedAt AS OrderDeletedAt
	, OOI.ID AS OrderItemId
	, OOI.DeletedAt AS OrderItemDeletedAt
	, OD.DeletedAt AS OrderDeliveryDeletedAt
	, ODI.DeletedAt AS OrderDeliveryItemDeletedAt
	, ODT.DeletedAt AS OrderDeliveryTimeDeletedAt
FROM
	Orders.Orders AS OO
LEFT JOIN	Orders.OrderItems AS OOI
	ON OOI.OrderID = OO.ID
LEFT JOIN Orders.Deliveries AS OD
	ON OD.OrderID = OO.ID
LEFT JOIN Orders.DeliveryItems AS ODI
	ON ODI.OrderItemID = OOI.ID
LEFT JOIN Orders.DeliveryTimes AS ODT
	ON ODT.DeliveryID = OD.ID
WHERE OO.ID = 2006;
	
DELETE Orders.Orders WHERE ID = 2006;
GO

SELECT
	OO.ID AS OrderId
	, OO.DeletedAt AS OrderDeletedAt
	, OOI.ID AS OrderItemId
	, OOI.DeletedAt AS OrderItemDeletedAt
	, OD.DeletedAt AS OrderDeliveryDeletedAt
	, ODI.DeletedAt AS OrderDeliveryItemDeletedAt
	, ODT.DeletedAt AS OrderDeliveryTimeDeletedAt
FROM
	Orders.Orders AS OO
LEFT JOIN	Orders.OrderItems AS OOI
	ON OOI.OrderID = OO.ID
LEFT JOIN Orders.Deliveries AS OD
	ON OD.OrderID = OO.ID
LEFT JOIN Orders.DeliveryItems AS ODI
	ON ODI.OrderItemID = OOI.ID
LEFT JOIN Orders.DeliveryTimes AS ODT
	ON ODT.DeliveryID = OD.ID
WHERE OO.ID = 2006;

UPDATE Orders.Orders SET DeletedAt = NULL WHERE ID = 2006;
UPDATE Orders.OrderItems SET DeletedAt = NULL WHERE OrderID = 2006;
GO

DELETE Orders.Orders WHERE ID = 2006;
GO
