USE FoodCourtReports;
GO

DROP PROCEDURE CreateMirrorTable;
GO

CREATE PROCEDURE CreateMirrorTable
    @SourceSchema NVARCHAR(128),
    @TableName NVARCHAR(128)
AS
BEGIN
    SET NOCOUNT ON;

    DECLARE @CHECK_QUERY NVARCHAR(MAX);
    DECLARE @CREATE_QUERY NVARCHAR(MAX);
	
	DECLARE @TargetSchema NVARCHAR(256);
	SET @TargetSchema = N'INGESTION';

    SET @CHECK_QUERY = N'SELECT @TableExists = 
							CASE 
								WHEN EXISTS (SELECT * FROM sys.tables WHERE name = @TableName AND SCHEMA_NAME(schema_id) = @TargetSchema) THEN 1 
								ELSE 0 
							END';

	DECLARE @TableExists BIT;
    EXEC sp_executesql @CHECK_QUERY, N'@TableName NVARCHAR(128), @TargetSchema NVARCHAR(256), @TableExists BIT OUTPUT', @TableName, @TargetSchema, @TableExists OUTPUT;

    IF @TableExists = 0
    BEGIN
        SET @CREATE_QUERY = N'SELECT * INTO FoodCourtReports.' + QUOTENAME(@TargetSchema) + '.' + QUOTENAME(@TableName) + 
                            N' FROM FoodCourt.' + QUOTENAME(@SourceSchema) + '.' + QUOTENAME(@TableName) + ' WHERE 1 = 0;';
		PRINT(@CREATE_QUERY);		
        EXEC sp_executesql @CREATE_QUERY;
    END
	ELSE
	BEGIN
		PRINT('Table ' + @TableName + ' already exsits');
	END
END;
GO


EXEC CreateMirrorTable
	@SourceSchema = 'Orders', 
	@TableName = 'Orders';

