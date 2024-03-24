USE FoodCourt
GO

BEGIN TRANSACTION T1
CREATE INDEX OrdersOrdersLocationIdIdx ON Orders.Orders(LocationId)
CREATE INDEX OrdersOrdersStatusIdx ON Orders.Orders(Status)
CREATE INDEX OrdersOrdersTypesIdx ON Orders.Orders(OrderType)
CREATE INDEX OrdersOrdersDeliveryAddressId ON Orders.Orders(DeliveryAddressId)
COMMIT TRANSACTION T1