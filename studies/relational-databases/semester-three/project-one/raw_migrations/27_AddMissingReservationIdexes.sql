USE FoodCourt
GO

BEGIN TRANSACTION T1
CREATE INDEX RestaurantsTablesLocationIdIdx ON Restaurants.Tables(LocationId)
CREATE INDEX RestaurantsReservationsCustomerIdIdx ON Restaurants.Reservations(CustomerId)
CREATE INDEX RestaurantsTableReservationsReservationIdIdx ON Restaurants.TableReservations(ReservationId)
CREATE INDEX RestaurantsTableReservationsTableIdIdx ON Restaurants.TableReservations(TableId)
COMMIT TRANSACTION T1