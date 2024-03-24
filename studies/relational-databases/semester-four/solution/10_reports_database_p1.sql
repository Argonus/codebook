/* Create FoodCourtReports Database */
IF NOT EXISTS (SELECT * FROM sys.databases WHERE name = N'FoodCourtReports')
BEGIN
    CREATE DATABASE FoodCourtReports;
END
ELSE
BEGIN
	PRINT('Database FoodCourtReports already exists')
END;
GO

/*Create ingestion schema */
USE FoodCourtReports
GO

IF NOT EXISTS (SELECT *  FROM sys.schemas WHERE name = N'INGESTION')
BEGIN
    EXEC('CREATE SCHEMA INGESTION;');
END
ELSE
BEGIN
	PRINT('Schema INGESTION already exists')
END;
GO
