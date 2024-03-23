CREATE LOGIN app_admin WITH PASSWORD = 'admin123';
CREATE LOGIN app_restaurants WITH PASSWORD = 'admin123';
CREATE LOGIN app_reports WITH PASSWORD = 'admin123';

USE FoodCourt;

/* Create Users */
CREATE USER app_admin FOR LOGIN app_admin;
CREATE USER app_restaurants FOR LOGIN app_restaurants;
CREATE USER app_reports FOR LOGIN app_reports;
GO

/* Assign Global Permissions */
/* Global Write */
EXEC sp_addrolemember 'db_datawriter', 'app_admin';
EXEC sp_addrolemember 'db_datawriter', 'app_restaurants';
/* Global Read */
EXEC sp_addrolemember 'db_datareader', 'app_admin';
EXEC sp_addrolemember 'db_datareader', 'app_restaurants';
EXEC sp_addrolemember 'db_datareader', 'app_reports';
GO
