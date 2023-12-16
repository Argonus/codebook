USE FoodCourt
GO

ALTER TABLE Orders.Orders
ALTER COLUMN DeliveryAddressID integer null

ALTER TABLE Orders.Orders DROP COLUMN Type;