USE FoodCourt
GO

CREATE TRIGGER TRG_AnonimizeAfterDelete
ON Clients.Customers
INSTEAD OF DELETE
AS
BEGIN
    SET NOCOUNT ON;

    DECLARE @ParentTableName NVARCHAR(256), @TableSchemaName NVARCHAR(256), @TableName NVARCHAR(256), @ColumnName NVARCHAR(256), @IdValue INT;
	DECLARE @CurrentDateTime DATETIME;
	DECLARE @RelationColumn NVARCHAR(256);
	DECLARE @QUERY NVARCHAR(MAX);

	SET @CurrentDateTime = GETDATE();

	/* Fetch Customer Id */
    DECLARE CustomerCursor CURSOR FOR SELECT ID FROM deleted;
    OPEN CustomerCursor;
    FETCH NEXT FROM CustomerCursor INTO @IdValue;
	CLOSE CustomerCursor;
    DEALLOCATE CustomerCursor;

	/* Soft Delete And Anonimizate Customer */
	UPDATE Clients.Customers SET DeletedAt = @CurrentDateTime FROM Clients.Customers WHERE ID = @IdValue;
	EXEC AnonimizeData @SchemaName = 'Clients', @TableName = 'Customers', @RelationColumn = 'ID', @IdValue = @IdValue;

	/* Soft Delete And Anonimizate Other Tables */ 
	DECLARE cursorLinkedTables CURSOR FOR
		SELECT SchemaName, TableName FROM dbo.GetLinkedTables('Customers', 'Clients');

    OPEN cursorLinkedTables;
	FETCH NEXT FROM cursorLinkedTables INTO @TableSchemaName, @TableName;

    WHILE @@FETCH_STATUS = 0
    BEGIN
	   PRINT 'Processing Table: ' + @TableName;

       SET @QUERY =  'UPDATE ' + QUOTENAME(@TableSchemaName) + '.' + QUOTENAME(@TableName) + ' SET DeletedAt = @CurrentDateTime ' +
					 'WHERE CustomerID = ' + CAST(@IdValue AS NVARCHAR(10)) + ';';
        EXEC sp_executesql @QUERY, N'@CurrentDateTime DATETIME', @CurrentDateTime;

		EXEC AnonimizeData @SchemaName = @TableSchemaName, @TableName = @TableName, @RelationColumn = 'CustomerID', @IdValue = @IdValue;
		FETCH NEXT FROM cursorLinkedTables INTO @TableSchemaName, @TableName;
    END

    CLOSE cursorLinkedTables;
    DEALLOCATE cursorLinkedTables;
END
GO

DELETE FROM Clients.Customers WHERE ID = 2006;