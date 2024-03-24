USE FoodCourt
GO

BEGIN TRANSACTION T1
CREATE INDEX StaffAddressesEmployees ON Staff.Addresses(EmployeeId)
CREATE INDEX StaffEmployeeLocationRolesEmployee ON Staff.EmployeeLocationRoles(EmployeeId)
CREATE INDEX StaffEmployeeLocationRolesRoles ON Staff.EmployeeLocationRoles(RoleId)
CREATE INDEX StaffEmployeeLocationRolesLocation ON Staff.EmployeeLocationRoles(LocationId)
CREATE INDEX StaffPhoneNumbersNumberType ON Staff.PhoneNumbers(NumberType)
COMMIT TRANSACTION T1