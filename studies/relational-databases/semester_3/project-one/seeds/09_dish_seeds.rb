# frozen_string_literal: true

class DishSeeds
  DISH_TYPES = ["Appetizer", "Main Course", "Dessert", "Drink"].freeze

  def initialize(db)
    @db = db
    @product_ids = @db["SELECT * FROM Resources.Products"].to_a.map { |product| product[:id] }
    @chef_ids = @db["
      SELECT e.id
      FROM Staff.Employees AS e
      INNER JOIN Staff.EmployeeLocationRoles AS er ON e.id = er.employeeId
      INNER JOIN Staff.Roles AS r ON er.roleId = r.id
      WHERE r.name = 'Executive Chef' OR r.name = 'Souse Chef'
    "].to_a.map { |employee| employee[:id] }
  end

  def run
    category_ids = create_categories()
    create_dish_types()

    100.times do
      category_id = category_ids.sample
      create_dish(category_id)
    end
  end

  def create_categories
    10.times.map do
      category_params = {
        name: Faker::Food.unique.ethnic_category[0..49],
      }
      data_set = @db["INSERT INTO Resources.Categories (name) VALUES (:name)", category_params]
      data_set.insert
    end
  end

  def create_dish_types
    DISH_TYPES.map do |dish_type|
      dish_type_params = {
        type: dish_type,
      }
      data_set = @db["INSERT INTO Resources.DishTypes (type) VALUES (:type)", dish_type_params]
      data_set.insert
    end
  end

  def create_dish(category_id)
    dish_params = {
      name: Faker::Food.dish[0..99],
      description: Faker::Food.description,
      categoryId: category_id,
      dishType: DISH_TYPES.sample,
      price: Faker::Commerce.price(range: 1..100.0),
      chefId: @chef_ids.sample,
    }
    data_set = @db["INSERT INTO Resources.Dishes (name, description, categoryId, dishType, price, chefId) VALUES (:name, :description, :categoryId, :dishType, :price, :chefId)", dish_params]
    data_set.insert
  end
end
