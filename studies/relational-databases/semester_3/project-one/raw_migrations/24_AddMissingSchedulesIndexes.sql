USE FoodCourt
GO

BEGIN TRANSACTION T1
CREATE INDEX StaffAssignmentDatesIdx ON Staff.EmployeeLocationRoles(ValidFrom, ValidTo DESC)
CREATE INDEX StaffSchedulesDates ON Staff.Schedules(StartDate, EndDate DESC)
COMMIT TRANSACTION T1