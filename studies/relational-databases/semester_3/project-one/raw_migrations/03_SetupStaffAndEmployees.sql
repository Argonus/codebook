USE FoodCourt
GO

CREATE SCHEMA Staff
GO

BEGIN TRANSACTION T1
/* Employee & Roles */
CREATE TABLE Staff.Employees (
    /* Table Data */
	ID INT IDENTITY(1,1) PRIMARY KEY,
	FirstName VARCHAR(255) NOT NULL,
    LastName VARCHAR(255) NOT NULL,
	BirthDate DATE NOT NULL,
	/* Timestamps */
	CreatedAt DATETIME2 DEFAULT GETDATE() NOT NULL,
    UpdatedAt DATETIME2 DEFAULT GETDATE() NOT NULL,
)

CREATE TABLE Staff.Roles (
    /* Table Data */
  	ID INT IDENTITY(1,1) PRIMARY KEY,
	Name VARCHAR(255) NOT NULL,
	/* Timestamps */
	CreatedAt DATETIME2 DEFAULT GETDATE() NOT NULL,
    UpdatedAt DATETIME2 DEFAULT GETDATE() NOT NULL,
)

CREATE TABLE Staff.EmployeeRoles(
  ID INT IDENTITY(1,1) PRIMARY KEY,
  /* Table Data */
  EmployeeId INT NOT NULL,
  RoleId INT NOT NULL,
  ValidFrom DATE NOT NULL,
  ValidTo Date NULL,
  /* Timestamps */
  CreatedAt DATETIME2 DEFAULT GETDATE() NOT NULL,
  UpdatedAt DATETIME2 DEFAULT GETDATE() NOT NULL,
  /* Foreign Keys */
  CONSTRAINT FK_Staff_EmployeeRoles_Employees FOREIGN KEY (EmployeeId) REFERENCES Staff.Employees (ID),
  CONSTRAINT FK_Staff_EmployeeRoles_Roles FOREIGN KEY (RoleId) REFERENCES Staff.Roles (ID),
  /* Constraints */
  CONSTRAINT CK_Staff_EmployeeRoles_ValidFromValidTo CHECK (ValidTo IS NULL OR ValidFrom <= ValidTo)
)
COMMIT TRANSACTION T1