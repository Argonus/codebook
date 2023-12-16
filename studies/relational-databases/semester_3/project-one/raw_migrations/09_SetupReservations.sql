USE FoodCourt
GO

BEGIN TRANSACTION T1
CREATE TABLE Restaurants.Tables (
	ID INT IDENTITY(1,1) PRIMARY KEY
	, LocationID SMALLINT NOT NULL
	, Description VARCHAR(50) NOT NULL
	, Seats TINYINT NOT NULL DEFAULT 0 
	/* Timestamps */
	, CreatedAt DATETIME2 DEFAULT GETDATE() NOT NULL
	, UpdatedAt DATETIME2 DEFAULT GETDATE() NOT NULL
	, DeletedAt DATETIME2 NULL
	/* Foreign Keys */
	, CONSTRAINT FK_Restaurants_Tables_Locations FOREIGN KEY (LocationID) REFERENCES Restaurants.Locations (ID)
)

CREATE TABLE Restaurants.Reservations (
	ID INT IDENTITY(1,1) PRIMARY KEY
	, CustomerID INT NOT NULL
	, ReservationDate DATE NOT NULL
	, ReservationHour TIME NOT NULL
	, ReservationTime SMALLINT NOT NULL DEFAULT 3600
	, Seats SMALLINT NOT NULL DEFAULT 1
	, Notes VARCHAR(MAX) NULL
	/* Timestamps */
	, CreatedAt DATETIME2 DEFAULT GETDATE() NOT NULL
	, UpdatedAt DATETIME2 DEFAULT GETDATE() NOT NULL
	, DeletedAt DATETIME2 NULL
	/* Foreign Keys */
	, CONSTRAINT FK_Restaurants_Reservations_Customers FOREIGN KEY (CustomerID) REFERENCES Clients.Customers (ID)
)

CREATE TABLE Restaurants.TableReservations (
	ID INT IDENTITY(1,1) PRIMARY KEY
	, TableID INT NOT NULL
	, ReservationID INT NOT NULL
	/* Timestamps */
	, CreatedAt DATETIME2 DEFAULT GETDATE() NOT NULL
	, UpdatedAt DATETIME2 DEFAULT GETDATE() NOT NULL
	, DeletedAt DATETIME2 NULL
	/* Foreign Keys */
	, CONSTRAINT FK_Restaurants_TableReservations_Tables FOREIGN KEY (TableID) REFERENCES Restaurants.Tables (ID)
	, CONSTRAINT FK_Restaurants_TableReservations_Reservations FOREIGN KEY (ReservationID) REFERENCES Restaurants.Reservations (ID)
)
GO
COMMIT TRANSACTION T1