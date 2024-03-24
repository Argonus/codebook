USE FoodCourt
GO

CREATE TABLE Restaurants.LocationMenus (
	ID INT IDENTITY(1,1) PRIMARY KEY
	, LocationID SMALLINT NOT NULL
	, MenuID INT NOT NULL
	, ValidFrom DATE NOT NULL
	, ValidTo DATE NULL
	 /* Timestamps */
    , CreatedAt DATETIME2 DEFAULT GETDATE() NOT NULL
	, UpdatedAt DATETIME2 DEFAULT GETDATE() NOT NULL
	/* Foreign Keys */
	, CONSTRAINT FK_Restaurants_LocationMenus_Menus FOREIGN KEY (MenuID) REFERENCES Resources.Menus (ID)
	, CONSTRAINT FK_Restaurants_LocationMenus_Locations FOREIGN KEY (LocationID) REFERENCES Restaurants.Locations (ID)
	/* Constraints */
	, CONSTRAINT CK_Restaurants_LocationMenus_Valid CHECK (ValidTo IS NULL OR ValidFrom <= ValidTo)
)