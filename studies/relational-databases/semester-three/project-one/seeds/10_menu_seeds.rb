# frozen_string_literal: true

class MenuSeeds
  def initialize(db)
    @db = db
    @location_ids = @db["SELECT * FROM Restaurants.Locations"].to_a.map { |location| location[:id] }
    @chef_ids = @db["
      SELECT e.id
      FROM Staff.Employees AS e
      INNER JOIN Staff.EmployeeLocationRoles AS er ON e.id = er.employeeId
      INNER JOIN Staff.Roles AS r ON er.roleId = r.id
      WHERE r.name = 'Executive Chef'
    "].to_a.map { |employee| employee[:id] }
  end

  def run
    %w[Summer Winter Spring Fall].each do |season|
      menu_id = create_menu(season)
      create_menu_locations(menu_id, season)
      create_menu_dishes(menu_id, season)
    end
  end

  def create_menu(season)
    menu_params = {
      name: "#{season} Menu",
      validFrom: season_start(season),
      validTo: season_end(season),
      executiveChefId: @chef_ids.sample,
    }
    data_set = @db["INSERT INTO Resources.Menus (name, validFrom, validTo, executiveChefId) VALUES (:name, :validFrom, :validTo, :executiveChefId)", menu_params]
    data_set.insert
  end

  def season_start(season)
    case season
    when "Winter" then Date.parse("2023-01-01")
    when "Summer" then Date.parse("2023-06-01")
    when "Spring" then Date.parse("2023-03-01")
    when "Fall" then Date.parse("2023-09-01")
    end
  end

  def season_end(season)
    case season
    when "Winter" then Date.parse("2023-02-28")
    when "Summer" then Date.parse("2023-08-31")
    when "Spring" then Date.parse("2023-05-31")
    when "Fall" then Date.parse("2023-12-31")
    end
  end

  def create_menu_locations(menu_id, season)
    @location_ids.each do |location_id|
      menu_location_params = {
        menuId: menu_id,
        locationId: location_id,
        validFrom: season_start(season),
        validTo: season_end(season),
      }
      data_set = @db["INSERT INTO Restaurants.LocationMenus (menuId, locationId, validFrom, validTo) VALUES (:menuId, :locationId, :validFrom, :validTo)", menu_location_params]
      data_set.insert
    end

    def create_menu_dishes(menu_id, season)
      10.times do
        menu_dish_params = {
          menuId: menu_id,
          dishId: @db["SELECT * FROM Resources.Dishes"].to_a.sample[:id],
          validFrom: season_start(season),
          validTo: season_end(season),
        }
        data_set = @db["INSERT INTO Resources.MenuDishes (menuId, dishId, validFrom, validTo) VALUES (:menuId, :dishId, :validFrom, :validTo)", menu_dish_params]
        data_set.insert
      end
    end
  end
end
