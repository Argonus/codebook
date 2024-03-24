USE FoodCourt
GO

SELECT
  SE.ID
  , SE.FirstName + ' ' + SE.LastName AS FullName
  , COUNT(OD.ID) AS DeliveriesCount
FROM
	Staff.Employees AS SE
INNER JOIN
	Staff.EmployeeLocationRoles AS SELR
	ON SE.ID = SELR.EmployeeID
INNER JOIN
	Staff.Roles AS SR
	ON SR.ID = SELR.RoleId
INNER JOIN
	Orders.Deliveries AS OD
	ON OD.EmployeeID = SE.ID
WHERE
	SR.Name = 'Delivery Driver'
	AND OD.Status = 'InProgress'
GROUP BY SE.ID, SE.FirstName, SE.LastName
HAVING COUNT(OD.ID) > 1
ORDER BY DeliveriesCount DESC
