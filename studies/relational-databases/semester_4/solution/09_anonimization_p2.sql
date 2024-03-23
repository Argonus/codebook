USE FoodCourt
GO

DROP PROCEDURE AddAnonizmiationPropertyToColumn;
GO

CREATE PROCEDURE AddAnonizmiationPropertyToColumn
    @SchemaName NVARCHAR(128),
    @TableName NVARCHAR(128),
    @ColumnName NVARCHAR(128)
AS
BEGIN
    SET NOCOUNT ON;

    IF EXISTS (
        SELECT * 
        FROM INFORMATION_SCHEMA.COLUMNS 
        WHERE TABLE_SCHEMA = @SchemaName 
        AND TABLE_NAME = @TableName 
        AND COLUMN_NAME = @ColumnName
    )
    BEGIN
		EXEC sp_addextendedproperty 
			@name = N'Anonimization', 
			@value = N'Yes', 
			@level0type = N'Schema', @level0name = @SchemaName, 
			@level1type = N'Table',  @level1name = @TableName, 
			@level2type = N'Column', @level2name = @ColumnName;

		PRINT('Anonimization Propery Added');
    END
    ELSE
    BEGIN
        PRINT('Column Does Not Exists');
    END
END;
GO

EXEC AddAnonizmiationPropertyToColumn @SchemaName = 'Clients', @TableName = 'Customers', @ColumnName = 'FirstName';
EXEC AddAnonizmiationPropertyToColumn @SchemaName = 'Clients', @TableName = 'Customers', @ColumnName = 'LastName';
EXEC AddAnonizmiationPropertyToColumn @SchemaName = 'Clients', @TableName = 'Customers', @ColumnName = 'Email';

EXEC AddAnonizmiationPropertyToColumn @SchemaName = 'Clients', @TableName = 'Addresses', @ColumnName = 'AddressOne';
EXEC AddAnonizmiationPropertyToColumn @SchemaName = 'Clients', @TableName = 'Addresses', @ColumnName = 'AddressTwo';
EXEC AddAnonizmiationPropertyToColumn @SchemaName = 'Clients', @TableName = 'Addresses', @ColumnName = 'Description';