CREATE FUNCTION dbo.GetLinkedTables 
(
    @TableName NVARCHAR(128),
    @SchemaName NVARCHAR(128)
)
RETURNS @Result TABLE 
(
    TableId NVARCHAR(256),
	SchemaName NVARCHAR(256),
    TableName NVARCHAR(256),
	ParentTableName NVARCHAR(256),
	ColumnName NVARCHAR(256),
	Level INT
)
AS
BEGIN
	WITH RecursiveFKs(ForeignKeyName, ParentObjectId, ReferenceObjectId, ParentColumnName, ReferencedColumnName, Level)
		AS
		(
			SELECT 
				FK.name AS ForeignKeyName,
				FK.parent_object_id AS ParentObjectId,
				FK.referenced_object_id AS ReferenceObjectId,
				COL_NAME(FKC.parent_object_id, FKC.parent_column_id) AS ParentColumnName,
				COL_NAME(FKC.referenced_object_id, FKC.referenced_column_id) AS ReferencedColumnName,
				1 AS Level
			FROM 
				sys.foreign_keys AS FK
			INNER JOIN	sys.foreign_key_columns AS FKC
				ON FK.object_id = FKC.constraint_object_id
			WHERE 
				OBJECT_NAME(FK.referenced_object_id) = @TableName 
				AND OBJECT_SCHEMA_NAME(FK.parent_object_id) = @SchemaName

			UNION ALL

			SELECT 
				FK.name AS ForeignKeyName,
				FK.parent_object_id AS ParentObjectId,
				FK.referenced_object_id AS ReferenceObjectId,
				COL_NAME(FKC.parent_object_id, FKC.parent_column_id) AS ParentColumnName,
				COL_NAME(FKC.referenced_object_id, FKC.referenced_column_id) AS ReferencedColumnName,
				R.Level + 1
			FROM 
				sys.foreign_keys AS FK
			INNER JOIN RecursiveFKs AS R
				ON FK.referenced_object_id = R.ParentObjectId
			INNER JOIN	sys.foreign_key_columns AS FKC
				ON FK.object_id = FKC.constraint_object_id
			WHERE
				OBJECT_NAME(fk.referenced_object_id) != @TableName
				AND OBJECT_SCHEMA_NAME(FK.parent_object_id) = @SchemaName
		)
	INSERT INTO @Result
	SELECT 
	DISTINCT 
		ParentObjectId AS TableId,
		OBJECT_SCHEMA_NAME(ParentObjectId) AS SchemaName,
		OBJECT_NAME(ParentObjectId) AS TableName,
		OBJECT_NAME(ReferenceObjectId) AS ParentTableName,
		ParentColumnName AS ColumnName,
		Level
	FROM RecursiveFKs
	ORDER BY Level
    RETURN;
END;
GO


CREATE TRIGGER TRG_SoftDeleteRelationships
ON Orders.Orders
INSTEAD OF DELETE
AS
BEGIN
    SET NOCOUNT ON;

	CREATE TABLE #ChangedRows(TableName NVARCHAR(128), ID INT);
    DECLARE @TableSchemaName NVARCHAR(128), @TableName NVARCHAR(128), @ParentTableName NVARCHAR(128), @ColumnName NVARCHAR(128);
	DECLARE @SchemaName NVARCHAR(128), @QUERY NVARCHAR(MAX), @CurrentDateTime DATETIME;

    SET @CurrentDateTime = GETDATE();
    SET @SchemaName = 'Orders'; 

    UPDATE Orders SET DeletedAt = @CurrentDateTime FROM Orders INNER JOIN deleted ON Orders.ID = deleted.ID;
	INSERT INTO #ChangedRows(TableName, ID) SELECT 'Orders', deleted.ID FROM deleted;

    DECLARE cursor_linkedTables CURSOR FOR
        SELECT SchemaName, TableName, ParentTableName, ColumnName FROM dbo.GetLinkedTables('Orders', @SchemaName);

    OPEN cursor_linkedTables;
    FETCH NEXT FROM cursor_linkedTables INTO @TableSchemaName, @TableName, @ParentTableName, @ColumnName;

    WHILE @@FETCH_STATUS = 0
    BEGIN
	   PRINT 'Aktualnie przetwarzana tabela: ' + @TableName;

       SET @QUERY =  'DECLARE @TempTable TABLE (ID INT);' +
                     'UPDATE ' + QUOTENAME(@TableSchemaName) + '.' + QUOTENAME(@TableName) + 
					 ' SET DeletedAt = @CurrentDateTime ' +
                     'OUTPUT INSERTED.ID INTO @TempTable ' +
					 'WHERE ' + QUOTENAME(@ColumnName) + ' IN (SELECT ID FROM #ChangedRows WHERE TableName = ''' + @ParentTableName + '''); ' +
					 'INSERT INTO #ChangedRows (TableName, ID) SELECT ''' + @TableName + ''', ID FROM @TempTable;';

        EXEC sp_executesql @QUERY, N'@CurrentDateTime DATETIME', @CurrentDateTime;
		FETCH NEXT FROM cursor_linkedTables INTO @TableSchemaName, @TableName, @ParentTableName, @ColumnName;
    END

	SELECT * FROM #ChangedRows;
	DROP TABLE #ChangedRows;
    CLOSE cursor_linkedTables;
    DEALLOCATE cursor_linkedTables;
END
