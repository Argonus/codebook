USE FoodCourt
GO

CREATE VIEW Restaurants.LocationDetails AS
WITH RestaurantPhoneNumbers AS (
	SELECT 
		LocationID
		, Number
		, row_number() over (partition by LocationID order by CreatedAt desc) as RowNum
	FROM
		Restaurants.PhoneNumbers
)
SELECT
	RL.ID AS LocationId
	, RL.Name AS LocationName
	, RA.AddressOne + ' ' + RA.AddressTwo AS LocationAddress
	, RA.ZipCode AS LocationZipCode
	, RA.Geom AS AddressGeom
FROM 
	Restaurants.Locations AS RL
INNER JOIN
	Restaurants.Addresses AS RA
	ON RL.Id = RA.LocationId AND RA.IsCurrent = 1
INNER JOIN 
	RestaurantPhoneNumbers AS RPN
	ON RPN.LocationID = RL.ID AND RPN.RowNum = 1;