USE FoodCourt

BEGIN TRANSACTION T1
CREATE TABLE Staff.Schedules (
    /* Table Data */
	ID INT IDENTITY(1,1) PRIMARY KEY
	, EmployeeId INT NOT NULL
	, LocationID SMALLINT NOT NULL
    , DayOfWeek TINYINT NOT NULL
	, RepeatFrequency TINYINT NOT NULL DEFAULT 1
	, StartDate Date NOT NULL
	, EndDate Date Null
	/* Timestamps */
	, CreatedAt DATETIME2 DEFAULT GETDATE() NOT NULL
    , UpdatedAt DATETIME2 DEFAULT GETDATE() NOT NULL
	/* Foreign Keys */
	, CONSTRAINT FK_Staff_Schedules_Employees FOREIGN KEY (EmployeeId) REFERENCES Staff.Employees (ID)
	, CONSTRAINT FK_Staff_Schedules_Locations FOREIGN KEY (LocationID) REFERENCES Restaurants.Locations (ID)
	/* Constraints */
	, CONSTRAINT CK_Staff_Schedules_Date CHECK (EndDate IS NULL OR StartDate <= EndDate)
	, CONSTRAINT CK_Staff_Schedules_RepeatFrequency CHECK (RepeatFrequency > 0 AND RepeatFrequency <= 4)
)

CREATE TABLE Staff.WorkingHours(
	/* Table Data */
	ID INT IDENTITY(1,1) PRIMARY KEY,
	ScheduleId INT NOT NULL,
	StartTime TIME NOT NULL,
	EndTime TIME NOT NULL,
	/* Timestamps */
	CreatedAt DATETIME2 DEFAULT GETDATE() NOT NULL,
    UpdatedAt DATETIME2 DEFAULT GETDATE() NOT NULL,
	/* Foreign Keys */
	CONSTRAINT FK_Staff_WorkingHours_Schedule FOREIGN KEY (ScheduleId) REFERENCES Staff.Schedules(ID),
	/* Constraints */
	CONSTRAINT CK_Staff_WorkingHours_Time CHECK (StartTime <= EndTime)
)
COMMIT TRANSACTION T1