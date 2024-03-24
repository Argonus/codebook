USE FoodCourt
GO

DROP PROCEDURE AnonimizeData
GO

CREATE PROCEDURE AnonimizeData
    @SchemaName NVARCHAR(128),
    @TableName NVARCHAR(128),
	@RelationColumn NVARCHAR(128),
    @IdValue INT
AS
BEGIN
    SET NOCOUNT ON;

    DECLARE @ColumnName NVARCHAR(128);
    DECLARE @QUERY NVARCHAR(MAX);

    DECLARE cursor_column CURSOR FOR 
		SELECT 
			col.name AS ColumnName
		FROM 
			sys.extended_properties AS SEXT
			INNER JOIN sys.columns AS COL
				ON SEXT.major_id = COL.object_id AND SEXT.minor_id = COL.column_id
			INNER JOIN sys.tables AS TBL
				ON COL.object_id = TBL.object_id
			WHERE 
				TBL.name = @TableName
				AND SCHEMA_NAME(TBL.schema_id) = @SchemaName
				AND SEXT.name = 'Anonimization'
				AND SEXT.Value = 'Yes';

    OPEN cursor_column;
    FETCH NEXT FROM cursor_column INTO @ColumnName;

    WHILE @@FETCH_STATUS = 0
    BEGIN
        PRINT('Processing Column ' + @ColumnName);
        SET @QUERY = 'UPDATE ' + QUOTENAME(@SchemaName) + '.' + QUOTENAME(@TableName) + 
                     ' SET ' + QUOTENAME(@ColumnName) + ' = ''Anonimized''' + 
                     ' WHERE ' + QUOTENAME(@ColumnName) + ' IS NOT NULL AND ' + @RelationColumn + ' = ' + CAST(@IdValue AS NVARCHAR(10)) + ';';
        EXEC sp_executesql @QUERY;

        FETCH NEXT FROM cursor_column INTO @ColumnName;
    END

    CLOSE cursor_column;
    DEALLOCATE cursor_column;
END;
GO

EXEC AnonimizeData @SchemaName = 'Clients', @TableName = 'Customers',  @RelationColumn = 'ID', @IdValue = 2005;
SELECT * FROM Clients.Customers WHERE ID = 2005;