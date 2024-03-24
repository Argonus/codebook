# frozen_string_literal: true

class LocationTableSeeds
  def initialize(db)
    @db = db
    @location_ids = @db["SELECT * FROM Restaurants.Locations"].to_a.map { |location| location[:id] }
  end

  def run
    @location_ids.each do |location_id|
      rand(2..10).times do
        create_table(location_id)
      end
    end
  end

  def create_table(location_id)
    table_params = {
      locationId: location_id,
      seats: rand(2..10),
      description: Faker::Restaurant.description[0..49],
    }
    data_set = @db["INSERT INTO Restaurants.Tables (locationId, seats, description) VALUES (:locationId, :seats, :description)", table_params]
    data_set.insert
  end
end
