USE FoodCourt
GO
/* Order Schema Validation */
/* Basic Constraints */
ALTER TABLE Orders.OrderItems ADD CONSTRAINT CHK_Quantity_NonNegative CHECK (Quantity >= 0);
ALTER TABLE Orders.DeliveryItems ADD CONSTRAINT CHK_Quantity_NonNegative CHECK (Quantity >= 0);
ALTER TABLE Orders.DeliveryTimes ADD CONSTRAINT CHK_DeliveryTime_NotFuture CHECK (DeliveredTime <= GETDATE());
GO

