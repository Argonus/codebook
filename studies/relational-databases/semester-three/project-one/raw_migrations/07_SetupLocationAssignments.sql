USE FoodCourt
GO

CREATE TABLE Staff.EmployeeLocationRoles (
	ID INT IDENTITY(1,1) PRIMARY KEY,
	EmployeeID INT NOT NULL,
	LocationID SMALLINT NOT NULL,
	RoleID INT NOT NULL,
	/* Datetime */
	ValidFrom DATE NOT NULL,
	ValidTo Date NULL,
	/* Timestamps */
    CreatedAt DATETIME2 DEFAULT GETDATE() NOT NULL,
	UpdatedAt DATETIME2 DEFAULT GETDATE() NOT NULL,
	/* Foreign Keys */
	CONSTRAINT FK_Staff_EmployeeLocationRoles_Employees FOREIGN KEY (EmployeeId) REFERENCES Staff.Employees (ID),
	CONSTRAINT FK_Staff_EmployeeLocationRoles_Locations FOREIGN KEY (LocationID) REFERENCES Restaurants.Locations (ID),
	CONSTRAINT FK_Staff_EmployeeLocationRoles_Roles FOREIGN KEY (RoleId) REFERENCES Staff.Roles (ID),
	/* Constraints */
	CONSTRAINT CK_Staff_EmployeeLocationRoles_ValidFromValidTo CHECK (ValidTo IS NULL OR ValidFrom <= ValidTo)
)
GO

CREATE TRIGGER Staff.TRG_CheckEmployeeLocationCity
ON Staff.EmployeeLocationRoles
AFTER INSERT, UPDATE
AS
BEGIN
  IF EXISTS (
    SELECT 
	  1
    FROM
      inserted AS ela
    JOIN Staff.Employees AS e
	  ON e.ID = ela.EmployeeID
    JOIN Staff.Addresses AS sa
	  ON sa.EmployeeID = e.ID
    JOIN Restaurants.Locations AS l
	  ON l.ID = ela.LocationID
	JOIN Restaurants.Addresses AS ra
	  ON ra.LocationID = l.ID
    WHERE
       sa.CityID != ra.CityID
    )
    BEGIN
        RAISERROR('Employee and location must be in the same city.', 16, 1);
        ROLLBACK TRANSACTION;
        RETURN;
    END
END;
GO

CREATE TRIGGER Staff.TRG_CheckEmployeeLocationAssigmentsSameCity
ON Staff.EmployeeLocationRoles
AFTER INSERT, UPDATE
AS
BEGIN
    IF EXISTS (
        SELECT 
			1
        FROM
            inserted AS NewAssignment
		JOIN Restaurants.Locations AS NewLocation
			ON NewLocation.ID = NewAssignment.LocationID
		JOIN Restaurants.Addresses AS NewAddress
			ON NewAddress.LocationID = NewLocation.ID
        JOIN Staff.EmployeeLocationRoles AS OldAssignemnt
			ON OldAssignemnt.EmployeeID = NewAssignment.EmployeeID
		JOIN Restaurants.Locations AS OldLocation
			ON OldAssignemnt.LocationID = OldLocation.ID
		JOIN Restaurants.Addresses AS OldAddress
			ON OldAddress.LocationID = OldLocation.ID
        WHERE
            NewAddress.CityID != OldAddress.CityID
    )
    BEGIN
        RAISERROR('The employee and the location must be in the same city.', 16, 1);
        ROLLBACK TRANSACTION;
        RETURN;
    END
END;
GO