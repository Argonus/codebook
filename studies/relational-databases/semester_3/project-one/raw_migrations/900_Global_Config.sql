USE FoodCourt
GO


DELETE FROM GlobalConfig.Cities
DELETE FROM GlobalConfig.States
GO

INSERT INTO GlobalConfig.States (Name)
VALUES
('Mazowieckie'),
('Malopolskie')

INSERT INTO GlobalConfig.Cities (Name, StateId)
SELECT
 'Warsaw' As Name
 , ID as StateId
FROM
  GlobalConfig.States
 WHERE 
  Name = 'Mazowieckie'

  INSERT INTO GlobalConfig.Cities (Name, StateId)
SELECT
 'Krakow' As Name
 , ID as StateId
FROM
  GlobalConfig.States
 WHERE 
  Name = 'Malopolskie'