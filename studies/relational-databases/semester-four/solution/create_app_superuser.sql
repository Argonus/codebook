/* Tworzenie Loginu */
CREATE LOGIN app_superuser WITH PASSWORD = 'admin123';

USE FoodCourt;

CREATE USER app_superuser FOR LOGIN app_superuser;
EXEC sp_addrolemember 'db_ddladmin', 'app_superuser';
EXEC sp_addrolemember 'db_datawriter', 'app_superuser';
EXEC sp_addrolemember 'db_datareader', 'app_superuser';

/* Testowanie Usera */
EXECUTE AS USER = 'app_superuser';
CREATE TABLE test(id int);
CREATE INDEX test_idx ON test(id)
INSERT INTO test VALUES(1);
SELECT * FROM test;
DROP TABLE test;
REVERT;
