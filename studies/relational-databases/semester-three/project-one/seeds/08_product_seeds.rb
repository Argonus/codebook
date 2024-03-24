# frozen_string_literal: true

class ProductSeeds
  def initialize(db)
    @db = db
  end

  def run
    allergen_isd = 5.times.map { create_allergen() }
    50.times do
      product_id = create_product()
      allergen_isd.sample(rand(0..3)).each do |allergen_id|
        create_product_allergen(product_id, allergen_id)
      end
    end
  end

  def create_allergen
    allergen_params = {
      name: Faker::Food.unique.allergen,
    }
    data_set = @db["INSERT INTO Resources.Allergens (name) VALUES (:name)", allergen_params]
    data_set.insert
  end

  def create_product
    product_params = {
      name: Faker::Food.unique.ingredient,
    }
    data_set = @db["INSERT INTO Resources.Products (name) VALUES (:name)", product_params]
    data_set.insert
  end

  def create_product_allergen(product_id, allergen_id)
    product_allergen_params = {
      productId: product_id,
      allergenId: allergen_id,
    }
    data_set = @db["INSERT INTO Resources.ProductAllergens (productId, allergenId) VALUES (:productId, :allergenId)", product_allergen_params]
    data_set.insert
  end
end
