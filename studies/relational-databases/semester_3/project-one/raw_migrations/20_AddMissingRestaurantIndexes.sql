USE FoodCourt
GO

BEGIN TRANSACTION T1
CREATE INDEX RestaurantsAddressesLocationIdx ON Restaurants.Addresses(LocationId)
CREATE INDEX RestaurantsAddressesCityIdx ON Restaurants.Addresses(CityId)
CREATE INDEX RestaurantsPhoneNumbersLocationIdx ON Restaurants.PhoneNumbers(LocationId)
CREATE INDEX RestaurantsSchedulesLocationIdx ON Restaurants.Schedules(LocationId)
CREATE INDEX RestaurantsOpeningHoursSchedulesIdx ON Restaurants.OpeningHours(ScheduleId)
COMMIT TRANSACTION T1


