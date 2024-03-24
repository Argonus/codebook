USE FoodCourtReports;
GO

CREATE OR ALTER PROCEDURE UpsertData
    @SourceSchema NVARCHAR(128),
    @SourceTableName NVARCHAR(128),
    @TargetSchema NVARCHAR(128),
    @TargetTableName NVARCHAR(128),
    @PrimaryKey NVARCHAR(128)
AS
BEGIN
    SET NOCOUNT ON;

    DECLARE @QUERY NVARCHAR(MAX);
    DECLARE @ColumnList NVARCHAR(MAX) = '';
    DECLARE @ColumnUpdateList NVARCHAR(MAX) = '';

    SELECT @ColumnList = STRING_AGG(QUOTENAME(column_name), ', ')
    FROM FoodCourt.INFORMATION_SCHEMA.COLUMNS
    WHERE TABLE_SCHEMA = @SourceSchema AND TABLE_NAME = @SourceTableName
    AND column_name <> @PrimaryKey;

    SELECT @ColumnUpdateList = STRING_AGG('Target.' + QUOTENAME(column_name) + ' = Source.' + QUOTENAME(column_name), ', ')
    FROM FoodCourt.INFORMATION_SCHEMA.COLUMNS
    WHERE TABLE_SCHEMA = @SourceSchema AND TABLE_NAME = @SourceTableName
    AND column_name <> @PrimaryKey;

    SET @QUERY = 'SET IDENTITY_INSERT FoodCourtReports.' + QUOTENAME(@TargetSchema) + '.' + QUOTENAME(@TargetTableName) + ' ON;' +
				 'MERGE INTO FoodCourtReports.' + QUOTENAME(@TargetSchema) + '.' + QUOTENAME(@TargetTableName) + ' AS Target ' +
                 'USING (SELECT * FROM FoodCourt.' + QUOTENAME(@SourceSchema) + '.' + QUOTENAME(@SourceTableName) + ') AS Source ' +
                 'ON Target.' + QUOTENAME(@PrimaryKey) + ' = Source.' + QUOTENAME(@PrimaryKey) + ' ' +
                 'WHEN MATCHED THEN UPDATE SET ' + @ColumnUpdateList + ' ' +
                 'WHEN NOT MATCHED BY TARGET THEN INSERT (' + @ColumnList + ', ' + QUOTENAME(@PrimaryKey) + ') VALUES (Source.' + @ColumnList + ', Source.' + QUOTENAME(@PrimaryKey) + ');';

	PRINT('List one')
	PRINT(@ColumnUpdateList)
	PRINT('List Two')
	PRINT(@ColumnList)
	PRINT(@QUERY)

    EXEC sp_executesql @QUERY;
END;
GO

EXEC UpsertData 
    @SourceSchema = 'Orders', 
    @SourceTableName = 'Orders',
    @TargetSchema = 'INGESTION', 
    @TargetTableName = 'Orders',
    @PrimaryKey = 'ID';

SELECT * FROM FoodCourtReports.INGESTION.Orders;

