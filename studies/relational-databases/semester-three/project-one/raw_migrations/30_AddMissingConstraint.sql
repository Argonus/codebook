USE FoodCourt
GO

ALTER TABLE Orders.DeliveryTimes
ADD CONSTRAINT FK_Orders_DeliveryTimes_Deliveries
FOREIGN KEY (DeliveryId) REFERENCES Orders.Deliveries(ID);

SELECT MAX(ID) FROM Restaurants.Locations;