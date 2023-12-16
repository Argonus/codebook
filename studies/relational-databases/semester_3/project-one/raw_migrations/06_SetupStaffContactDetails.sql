USE FoodCourt
GO

BEGIN TRANSACTION T1
CREATE TABLE Staff.Addresses (
  ID INT IDENTITY(1,1) PRIMARY KEY,
  EmployeeID INT NOT NULL,
  AddressOne VARCHAR(255) NOT NULL,
  AddressTwo VARCHAR(255) NULL,
  ZipCode CHAR(5) NOT NULL,
  CityId INT NOT NULL,
  /* Timestamps */
  CreatedAt DATETIME2 DEFAULT GETDATE() NOT NULL,
  UpdatedAt DATETIME2 DEFAULT GETDATE() NOT NULL,
  /* Foreign Keys */
  CONSTRAINT FK_Staff_Addresses_Locations FOREIGN KEY (EmployeeID) REFERENCES Staff.Employees (ID),
  CONSTRAINT FK_Staff_Addresses_Cities FOREIGN KEY (CityId) REFERENCES GlobalConfig.Cities (ID),
)

/* Lookup Table */
CREATE TABLE Staff.PhoneNumberTypes (
  Type VARCHAR(15) PRIMARY KEY,
)

CREATE TABLE Staff.PhoneNumbers (
  ID INT IDENTITY(1,1) PRIMARY KEY,
  NumberType VARCHAR(15) NOT NULL,
  EmployeeID INT NOT NULL,
  Number VARCHAR(15) NOT NULL,
  Description VARCHAR(255) NULL,
  /* Timestamps */
  CreatedAt DATETIME2 DEFAULT GETDATE() NOT NULL,
  UpdatedAt DATETIME2 DEFAULT GETDATE() NOT NULL,
  /* Foreign Keys */
  CONSTRAINT FK_Staff_PhoneNumebers_Locations FOREIGN KEY (EmployeeID) REFERENCES Staff.Employees (ID),
  CONSTRAINT FK_Staff_PhoneNumebers_PhoneNumberTypes FOREIGN KEY (NumberType) REFERENCES Staff.PhoneNumberTypes (Type),
)
COMMIT TRANSACTION T1