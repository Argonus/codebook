USE FoodCourt
GO

/* Add Deleted At Field to Clients Tables */
DECLARE @TableName NVARCHAR(256);
DECLARE @ColumnName NVARCHAR(256);
DECLARE @Sql NVARCHAR(MAX);
SET @ColumnName = 'DeletedAt'

-- Kursor do iteracji przez wszystkie tabele w schemacie 'Clients'
DECLARE cursor_table CURSOR FOR 
    SELECT t.name
    FROM sys.tables AS t
    INNER JOIN sys.schemas AS s ON t.schema_id = s.schema_id
    WHERE s.name = 'Clients';

OPEN cursor_table;
FETCH NEXT FROM cursor_table INTO @TableName;

WHILE @@FETCH_STATUS = 0
BEGIN
    IF NOT EXISTS (
        SELECT * 
        FROM sys.columns AS c
        INNER JOIN sys.tables AS t ON c.object_id = t.object_id
        INNER JOIN sys.schemas AS s ON t.schema_id = s.schema_id
        WHERE s.name = 'Clients' 
        AND t.name = @TableName 
        AND c.name = @ColumnName
    )
    BEGIN
		PRINT('Processing Table ' + @TableName);
        SET @Sql = N'ALTER TABLE Clients.' + QUOTENAME(@TableName) + N' ADD ' + @ColumnName + ' DATETIME NULL;';
        EXEC sp_executesql @Sql;
    END

    FETCH NEXT FROM cursor_table INTO @TableName;
END

CLOSE cursor_table;
DEALLOCATE cursor_table;
GO
