USE FoodCourt
GO

CREATE SCHEMA Resources
GO

BEGIN TRANSACTION T1
CREATE TABLE Resources.Menus (
	ID INT IDENTITY(1,1) PRIMARY KEY
	, Name VARCHAR(50)
	, ValidFrom DATE NOT NULL
	, ValidTo DATE NULL
	, ExecutiveChefID INT NOT NULL
	/* Timestamps */
	, CreatedAt DATETIME2 DEFAULT GETDATE() NOT NULL
	, UpdatedAt DATETIME2 DEFAULT GETDATE() NOT NULL
	/* Foreign Keys */
	, CONSTRAINT FK_Resources_Menu_Employees FOREIGN KEY (ExecutiveChefID) REFERENCES Staff.Employees (ID)
	/* Constraints */
	, CONSTRAINT CK_Resources_Dishes_Valid CHECK (ValidTo IS NULL OR ValidFrom <= ValidTo)
)

CREATE TABLE Resources.Categories (
	ID INT IDENTITY(1,1) PRIMARY KEY
	, Name VARCHAR(50)
	/* Timestamps */
	, CreatedAt DATETIME2 DEFAULT GETDATE() NOT NULL
	, UpdatedAt DATETIME2 DEFAULT GETDATE() NOT NULL
)
CREATE TABLE Resources.Dishes (
	ID INT IDENTITY(1,1) PRIMARY KEY
	, Name VARCHAR(100) NOT NULL
	, Description VARCHAR(MAX) NOT NULL
	, Price SMALLMONEY NOT NULL
	, CategoryID INT NOT NULL
	, ChefID INT NOT NULL
	/* Timestamps */
	, CreatedAt DATETIME2 DEFAULT GETDATE() NOT NULL
	, UpdatedAt DATETIME2 DEFAULT GETDATE() NOT NULL
	/* Foreign Keys */
	, CONSTRAINT FK_Resources_Dishes_Employees FOREIGN KEY (ChefID) REFERENCES Staff.Employees (ID)
	, CONSTRAINT FK_Resources_Dishes_Categories FOREIGN KEY (CategoryID) REFERENCES Resources.Categories (ID)

)

CREATE TABLE Resources.MenuDishes (
	ID INT IDENTITY(1,1) PRIMARY KEY
	, MenuID INT NOT NULL
	, DishID INT NOT NULL
	, ValidFrom DATE NOT NULL
	, ValidTo DATE NULL
	/* Timestamps */
	, CreatedAt DATETIME2 DEFAULT GETDATE() NOT NULL
	, UpdatedAt DATETIME2 DEFAULT GETDATE() NOT NULL
	/* Foreign Keys */
	, CONSTRAINT FK_Resources_MenuDishes_Menu FOREIGN KEY (MenuID) REFERENCES Resources.Menus (ID)
	, CONSTRAINT FK_Resources_MenuDishes_Dish FOREIGN KEY (DishID) REFERENCES Resources.Dishes (ID)
	/* Constraints */
	, CONSTRAINT CK_Resources_MenuDishes_Valid CHECK (ValidTo IS NULL OR ValidFrom <= ValidTo)
)

CREATE TABLE Resources.Products (
	ID INT IDENTITY(1,1) PRIMARY KEY
	, Name VARCHAR(255) NOT NULL
	/* Timestamps */
	, CreatedAt DATETIME2 DEFAULT GETDATE() NOT NULL
	, UpdatedAt DATETIME2 DEFAULT GETDATE() NOT NULL
)

CREATE TABLE Resources.DishProducts (
	ID INT IDENTITY(1,1) PRIMARY KEY
	, DishID INT NOT NULL
	, ProductID INT NOT NULL
	/* Timestamps */
	, CreatedAt DATETIME2 DEFAULT GETDATE() NOT NULL
	, UpdatedAt DATETIME2 DEFAULT GETDATE() NOT NULL
	/* Foreign Keys */
	, CONSTRAINT FK_Resources_DishProducts_Dish FOREIGN KEY (DishID) REFERENCES Resources.Dishes (ID)
	, CONSTRAINT FK_Resources_DishProducts_Product FOREIGN KEY (ProductID) REFERENCES Resources.Products (ID)
)

CREATE TABLE Resources.Allergens (
	ID INT IDENTITY(1,1) PRIMARY KEY
	, Name VARCHAR(255) NOT NULL
	/* Timestamps */
	, CreatedAt DATETIME2 DEFAULT GETDATE() NOT NULL
	, UpdatedAt DATETIME2 DEFAULT GETDATE() NOT NULL
)

CREATE TABLE Resources.ProductAllergens (
	ID INT IDENTITY(1,1) PRIMARY KEY
	, ProductID INT NOT NULL
	, AllergenID INT NOT NULL
	/* Timestamps */
	, CreatedAt DATETIME2 DEFAULT GETDATE() NOT NULL
	, UpdatedAt DATETIME2 DEFAULT GETDATE() NOT NULL
	/* Foreign Keys */
	, CONSTRAINT FK_Resources_ProductAllergens_Product FOREIGN KEY (ProductID) REFERENCES Resources.Products (ID)
	, CONSTRAINT FK_Resources_ProductAllergens_Allergen FOREIGN KEY (AllergenID) REFERENCES Resources.Allergens (ID)
)
GO
COMMIT TRANSACTION T1