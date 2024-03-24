/* Create Title Rule */
USE FoodCourt
GO

CREATE RULE TITLE_RULE AS @title IN ('Mr', 'Mrs', 'Miss');
GO

/* Add Title To Tables */
IF NOT EXISTS (SELECT * FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_SCHEMA = 'Staff' AND TABLE_NAME = 'Employees' AND COLUMN_NAME = 'Title')
BEGIN
    ALTER TABLE Staff.Employees
    ADD Title NVARCHAR(5) NULL;
	EXEC sp_bindrule 'TITLE_RULE', 'Staff.Employees.Title';
END
ELSE
BEGIN
    PRINT('Column Title already exists');
END;

IF NOT EXISTS (SELECT * FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_SCHEMA = 'Clients' AND TABLE_NAME = 'Customers' AND COLUMN_NAME = 'Title')
BEGIN
    ALTER TABLE Clients.Customers
    ADD Title NVARCHAR(5) NULL;
	EXEC sp_bindrule 'TITLE_RULE', 'Clients.Customers.Title';
END
ELSE
BEGIN
    PRINT('Column Title already exists');
END;

SELECT * FROM Staff.Employees;
UPDATE Staff.Employees SET Title = 'Mr' WHERE ID = 361;
GO

UPDATE Staff.Employees SET Title = 'Mrr' WHERE ID = 373;
GO