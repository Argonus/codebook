USE FoodCourt
GO

CREATE VIEW Restaurants.ActiveEmployees AS
SELECT
	SE.ID AS EmployeeId
	, SE.FirstName + ' ' + SE.LastName AS EmployeeFullName
	, SR.Name AS Role
	, SELR.ValidTo AS WorkingUntill
	, SELR.LocationId AS LocationId
FROM Staff.Employees AS SE
INNER JOIN Staff.EmployeeLocationRoles AS SELR
	ON SE.Id = SELR.EmployeeId
INNER JOIN Staff.Roles AS SR
	ON SELR.RoleId = SR.Id
WHERE 
	SELR.ValidFrom <= CAST(GETDATE() AS Date) AND SELR.ValidTo >= CAST(GETDATE() AS Date);