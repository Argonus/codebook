/* Create user developer */
CREATE LOGIN user_tomek WITH PASSWORD = 'user123' MUST_CHANGE, CHECK_POLICY = ON, CHECK_EXPIRATION = ON;
CREATE USER user_tomek FOR LOGIN user_tomek;
CREATE LOGIN user_anna WITH PASSWORD = 'user123' MUST_CHANGE, CHECK_POLICY = ON, CHECK_EXPIRATION = ON;
CREATE USER user_anna FOR LOGIN user_anna;

CREATE ROLE user_developer_role;
CREATE ROLE user_maintenance_role;

/*Maintenance Role */
EXEC sp_addrolemember 'db_datareader', 'user_anna';
EXEC sp_addrolemember 'db_datawriter', 'user_anna';

/*Developer Role*/
EXEC sp_addrolemember 'db_datareader', 'user_tomek';
DENY SELECT ON Clients.Addresses TO user_developer_role;
DENY SELECT ON Clients.Customers([FirstName]) TO user_developer_role;
DENY SELECT ON Clients.Customers([PhoneNumber]) TO user_developer_role;
DENY SELECT ON Clients.Customers([Email]) TO user_developer_role;
DENY SELECT ON Staff.Addresses TO user_developer_role;
DENY SELECT ON Staff.PhoneNumbers TO user_developer_role;
DENY SELECT ON Staff.Employees([LastName]) TO user_developer_role;



