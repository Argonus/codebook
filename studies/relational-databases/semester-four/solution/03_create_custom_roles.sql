/* Create Login */
CREATE LOGIN app_orders WITH PASSWORD = 'admin123';

USE FoodCourt;

/* Create Users */
CREATE USER app_orders FOR LOGIN app_orders;
GO

/* Assign Global Permissions */
EXEC sp_addrolemember 'db_datareader', 'app_orders';
GO

/* Schema Permissions */
CREATE ROLE app_orders_role;
GO

/* app orders role */
GRANT INSERT, UPDATE, DELETE ON SCHEMA::Clients TO app_orders_role;
GRANT INSERT, UPDATE, DELETE ON SCHEMA::Orders TO app_orders_role;
GO

ALTER ROLE app_orders_role ADD MEMBER app_orders;
GO