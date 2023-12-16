USE FoodCourt 
GO

CREATE SCHEMA Restaurants
GO

BEGIN TRANSACTION T1
CREATE TABLE Restaurants.Locations (
    /* Table Data */
	ID SMALLINT IDENTITY(1,1) PRIMARY KEY,
	Name CHAR(255) NOT NULL,
    Slug VARCHAR(255) NOT NULL,
	/* Timestamps */
	CreatedAt DATETIME2 DEFAULT GETDATE() NOT NULL,
    UpdatedAt DATETIME2 DEFAULT GETDATE() NOT NULL,
)

CREATE TABLE Restaurants.Schedules (
    /* Table Data */
	ID INT IDENTITY(1,1) PRIMARY KEY,
	LocationId SMALLINT NOT NULL,
    DayOfWeek TINYINT NOT NULL,
	StartDate Date NOT NULL,
	EndDate Date Null,
	/* Timestamps */
	CreatedAt DATETIME2 DEFAULT GETDATE() NOT NULL,
    UpdatedAt DATETIME2 DEFAULT GETDATE() NOT NULL,
	/* Foreign Keys */
	CONSTRAINT FK_Restaurants_Schedules_Locations FOREIGN KEY (LocationID) REFERENCES Restaurants.Locations (ID),
	/* Constraints */
	CONSTRAINT CK_Restaurants_LocationsSchedules_Date CHECK (EndDate IS NULL OR StartDate <= EndDate)
)

CREATE TABLE Restaurants.OpeningHours(
	/* Table Data */
	ID INT IDENTITY(1,1) PRIMARY KEY,
	ScheduleId INT NOT NULL,
	StartTime TIME NOT NULL,
	EndTime TIME NOT NULL,
	/* Timestamps */
	CreatedAt DATETIME2 DEFAULT GETDATE() NOT NULL,
    UpdatedAt DATETIME2 DEFAULT GETDATE() NOT NULL,
	/* Foreign Keys */
	CONSTRAINT FK_Restaurants_OpeningHours_Schedule FOREIGN KEY (ScheduleId) REFERENCES Restaurants.Schedules(ID),
	/* Constraints */
	CONSTRAINT CK_Restaurants_LocationsSchedulesOpeningHours_Time CHECK (StartTime <= EndTime)
)
COMMIT TRANSACTION T1