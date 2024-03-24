USE FoodCourt
GO

BEGIN TRANSACTION T1
CREATE TABLE Restaurants.Addresses (
  /* Table Data */
  ID INT IDENTITY(1,1) PRIMARY KEY,
  LocationID SMALLINT NOT NULL,
  AddressOne VARCHAR(255) NOT NULL,
  AddressTwo VARCHAR(255) NULL,
  Longitude DECIMAL(9,6) NOT NULL,
  Latitude DECIMAL(8,6) NOT NULL,
  Geom AS geography::Point(Latitude, Longitude, 4326) PERSISTED,
  ZipCode CHAR(5) NOT NULL,
  CityId INT NOT NULL,
  /* Timestamps */
  CreatedAt DATETIME2 DEFAULT GETDATE() NOT NULL,
  UpdatedAt DATETIME2 DEFAULT GETDATE() NOT NULL,
  /* Foreign Keys */
  CONSTRAINT FK_Restaurants_Addresses_Locations FOREIGN KEY (LocationID) REFERENCES Restaurants.Locations (ID),
  CONSTRAINT FK_Restaurants_Addresses_Cities FOREIGN KEY (CityId) REFERENCES GlobalConfig.Cities (ID),
)
CREATE SPATIAL INDEX SIndx_Restaurants_Addresses_Geom ON Restaurants.Addresses(Geom);

CREATE TABLE Restaurants.PhoneNumbers (
  ID INT IDENTITY(1,1) PRIMARY KEY,
  LocationID SMALLINT NOT NULL,
  Number VARCHAR(15) NOT NULL,
  Description VARCHAR(255) NOT NULL,
  /* Timestamps */
  CreatedAt DATETIME2 DEFAULT GETDATE() NOT NULL,
  UpdatedAt DATETIME2 DEFAULT GETDATE() NOT NULL,
  /* Foreign Keys */
  CONSTRAINT FK_Restaurants_PhoneNumbers_Locations FOREIGN KEY (LocationID) REFERENCES Restaurants.Locations (ID),
)
COMMIT TRANSACTION T1