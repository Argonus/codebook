USE FoodCourt
GO

SELECT E.ID, E.FirstName, E.LastName
FROM Staff.Employees AS E
INNER JOIN Staff.EmployeeLocationRoles AS SELR
    ON E.ID = SELR.EmployeeID
WHERE SELR.LocationID = 137

EXCEPT

SELECT E.ID, E.FirstName, E.LastName
FROM Staff.Employees AS E
INNER JOIN Resources.Dishes AS RD
    ON E.ID = RD.ChefID
