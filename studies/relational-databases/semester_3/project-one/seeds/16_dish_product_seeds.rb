# frozen_string_literal: true

class DishProductsSeeds
  def initialize(db)
    @db = db
    @product_ids = @db["SELECT * FROM Resources.Products"].to_a.map { |product| product[:id] }
    @dish_ids = @db["SELECT * FROM Resources.Dishes"].to_a.map { |dish| dish[:id] }
  end

  def run
    @dish_ids.each do |dish_id|
      products_count = rand(1..5)
      @product_ids.sample(products_count).each do |product_id|
        create_dish_product(dish_id, product_id)
      end
    end
  end

  def create_dish_product(dish_id, product_id)
    params = {
      dishId: dish_id,
      productId: product_id,
    }
    data_set = @db["INSERT INTO Resources.DishProducts (DishId, ProductId) VALUES (:dishId, :productId)", params]
    data_set.insert
  end
end