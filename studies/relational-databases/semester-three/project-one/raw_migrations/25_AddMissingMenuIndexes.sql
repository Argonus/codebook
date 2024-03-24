USE FoodCourt
GO

BEGIN TRANSACTION T1
CREATE INDEX ResourcesDishesCategoryIdx ON Resources.Dishes(CategoryId)
CREATE INDEX ResourcesDishesDishTypeIdx ON Resources.Dishes(DishType)
CREATE INDEX ResourcesDishesChefIdIdx ON Resources.Dishes(ChefId)
CREATE INDEX ResourcesMenuDishesDishIdIdx ON Resources.MenuDishes(DishId)
CREATE INDEX ResourcesMenuDishesMenuIdIdx ON Resources.MenuDishes(MenuId)
CREATE INDEX ResourcesMenusExecutiveChefIdIdx ON Resources.Menus(ExecutiveChefId)
CREATE INDEX ResourcesMenusValidDateIdx ON Resources.Menus(ValidFrom, ValidTo DESC)
CREATE INDEX RestauransLocationMenusDateIdx ON Restaurants.LocationMenus(ValidFrom, ValidTo DESC)
CREATE INDEX RestauransLocationMenusLocationId ON Restaurants.LocationMenus(LocationId)
CREATE INDEX RestauransLocationMenusMenuId ON Restaurants.LocationMenus(MenuId)
COMMIT TRANSACTION T1