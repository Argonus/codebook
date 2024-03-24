USE FoodCourt
GO

BEGIN TRANSACTION T1
CREATE INDEX ClientsAddressesCustomerIdIdx ON Clients.Addresses(CustomerId)
CREATE INDEX ClientsAddressesCityIdIdx ON Clients.Addresses(CityId)
COMMIT TRANSACTION T1