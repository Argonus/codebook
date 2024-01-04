CREATE TRIGGER TRG_CheckQuantity ON Orders.DeliveryItems
AFTER INSERT, UPDATE
AS
BEGIN
    SET NOCOUNT ON;

    IF EXISTS (
        SELECT 1
        FROM Inserted AS I
        INNER JOIN Orders.OrderItems AS OOI
		ON I.OrderItemID = OOI.ID
        WHERE I.Quantity > OOI.Quantity
    )
    BEGIN
		/* msg, severity, state */
        RAISERROR('Quantity in DeliveryItem cannot exceed Quantity in OrderItem', 10, 1);
        ROLLBACK TRANSACTION;
        RETURN;
    END
END
