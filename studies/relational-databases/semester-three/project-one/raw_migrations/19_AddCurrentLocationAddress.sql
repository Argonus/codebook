USE FoodCourt
GO

ALTER TABLE Restaurants.Addresses 
ADD IsCurrent BIT NOT NULL DEFAULT 0

DROP INDEX Restaurants_Current_Address ON Restaurants.Addresses;
CREATE UNIQUE INDEX Restaurants_Current_Address ON Restaurants.Addresses(LocationId, IsCurrent) WHERE IsCurrent = 1;
