USE FoodCourt
GO

CREATE SCHEMA Orders
GO

BEGIN TRANSACTION T1
CREATE TABLE Orders.OrderStatuses (
	Status VARCHAR(25) PRIMARY KEY
	/* Timestamps */
    , CreatedAt DATETIME2 DEFAULT GETDATE() NOT NULL
	, UpdatedAt DATETIME2 DEFAULT GETDATE() NOT NULL
)

CREATE TABLE Orders.OrderTypes (
	Type VARCHAR(25) PRIMARY KEY
	/* Timestamps */
    , CreatedAt DATETIME2 DEFAULT GETDATE() NOT NULL
	, UpdatedAt DATETIME2 DEFAULT GETDATE() NOT NULL
)

CREATE TABLE Orders.Orders (
	ID INT IDENTITY(1,1) PRIMARY KEY
	, LocationID SMALLINT NOT NULL
	, OrderDateTime DATETIME NOT NULL
	, DeliveryDateTime DATETIME NULL
	, Status VARCHAR(25) NOT NULL
	, Type VARCHAR(25) NOT NULL
	, DeliveryAddressID INT NOT NULL
	/* Timestamps */
	, CreatedAt DATETIME2 DEFAULT GETDATE() NOT NULL
	, UpdatedAt DATETIME2 DEFAULT GETDATE() NOT NULL
	, DeletedAt DATETIME2 NULL
	/* Foreign Keys */
	, CONSTRAINT FK_Orders_Orders_OrderStatuses FOREIGN KEY (Status) REFERENCES Orders.OrderStatuses (Status)
	, CONSTRAINT FK_Orders_Orders_Locations FOREIGN KEY (LocationID) REFERENCES Restaurants.Locations (ID)
	, CONSTRAINT FK_Orders_Orders_DeliveryAddresses FOREIGN KEY (DeliveryAddressID) REFERENCES Clients.Addresses (ID)
	/* Constraints */
	, CONSTRAINT CK_Orders_Orders_DeliveryTime CHECK (DeliveryDateTime IS NULL OR DeliveryDateTime > OrderDateTime)
)

CREATE TABLE Orders.OrderItems (
	ID INT IDENTITY(1,1) PRIMARY KEY
	, OrderID INT NOT NULL
	, DishID INT NOT NULL
	, Quantity SMALLINT NOT NULL
	/* Timestamps */
	, CreatedAt DATETIME2 DEFAULT GETDATE() NOT NULL
	, UpdatedAt DATETIME2 DEFAULT GETDATE() NOT NULL
	, DeletedAt DATETIME2 NULL
	/* Foreign Keys */
	, CONSTRAINT FK_Orders_OrderItems_Orders FOREIGN KEY (OrderID) REFERENCES Orders.Orders (ID)
	, CONSTRAINT FK_Orders_OrderItems_Dishes FOREIGN KEY (DishID) REFERENCES Resources.Dishes (ID)
	/* Constraints */
	, CONSTRAINT CK_Orders_DeliveryDishes_Quantity CHECK (Quantity > 0)
)


CREATE TABLE Orders.DeliveryStatuses (
	Status VARCHAR(25) PRIMARY KEY
	/* Timestamps */
    , CreatedAt DATETIME2 DEFAULT GETDATE() NOT NULL
	, UpdatedAt DATETIME2 DEFAULT GETDATE() NOT NULL
)

CREATE TABLE Orders.Deliveries (
	ID INT IDENTITY(1,1) PRIMARY KEY
	, OrderID INT NOT NULL
	, EmployeeID INT NOT NULL
	, Status VARCHAR(25) NOT NULL
	/* Timestamps */
	, CreatedAt DATETIME2 DEFAULT GETDATE() NOT NULL
	, UpdatedAt DATETIME2 DEFAULT GETDATE() NOT NULL
	, DeletedAt DATETIME2 NULL
	/* Foreign Keys */
	, CONSTRAINT FK_Orders_Deliveries_Orders FOREIGN KEY (OrderID) REFERENCES Orders.Orders (ID)
	, CONSTRAINT FK_Orders_Deliveries_Employees FOREIGN KEY (EmployeeID) REFERENCES Staff.Employees (ID)
	, CONSTRAINT FK_Orders_Deliveries_DeliveryStatuses FOREIGN KEY (Status) REFERENCES Orders.DeliveryStatuses (Status)
)

CREATE TABLE Orders.DeliveryItems (
	ID INT IDENTITY(1,1) PRIMARY KEY
	, DeliveryID INT NOT NULL
	, OrderItemID INT NOT NULL
	, Quantity SMALLINT NOT NULL
	/* Timestamps */
	, CreatedAt DATETIME2 DEFAULT GETDATE() NOT NULL
	, UpdatedAt DATETIME2 DEFAULT GETDATE() NOT NULL
	, DeletedAt DATETIME2 NULL
	/* Foreign Keys */
	, CONSTRAINT FK_Orders_DeliveryDishes_Deliveries FOREIGN KEY (DeliveryID) REFERENCES Orders.Deliveries (ID)
	, CONSTRAINT FK_Orders_DeliveryDishes_DishIDes FOREIGN KEY (OrderItemID) REFERENCES Orders.OrderItems (ID)
	/* Constraints */
	, CONSTRAINT CK_Orders_DeliveryItems_Quantity CHECK (Quantity > 0)
)
COMMIT TRANSACTION T1