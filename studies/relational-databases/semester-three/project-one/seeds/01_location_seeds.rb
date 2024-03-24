# frozen_string_literal: true
class LocationSeeds
  def initialize(db)
    @db = db
  end

  def run
    city_ids = @db["SELECT * FROM GlobalConfig.Cities"].to_a.map { |city| city[:id] }
    city_ids.each do |city_id|
      10.times { |n| create_location(n, city_id) }
    end
  end

  def create_location(n, city_id)
    name = Faker::Company.name
    location_params = { name: name, slug: name.parameterize }
    location_id = @db["INSERT INTO Restaurants.Locations (name, slug) VALUES (:name, :slug)", location_params].insert

    [0, 1].each do |active|
      address_params = {
        addressOne: Faker::Address.street_address.to_s,
        addressTwo: Faker::Address.secondary_address.to_s,
        longitude: Faker::Address.longitude.round(6),
        latitude: Faker::Address.latitude.round(6),
        zipCode: Faker::Address.zip_code[0..4],
        city_id: city_id,
        location_id: location_id,
        active: active,
      }
      data_set = @db["INSERT INTO Restaurants.Addresses (addressOne, addressTwo, longitude, latitude, zipCode, cityId, locationId, isCurrent) VALUES (:addressOne, :addressTwo, :longitude, :latitude, :zipCode, :city_id, :location_id, :active)", address_params]
      data_set.insert
    end

    phone_number_params = {
      locationId: location_id,
      number: Faker::PhoneNumber.cell_phone_in_e164[0..14],
      description: Faker::PhoneNumber.extension,
    }
    data_set = @db["INSERT INTO Restaurants.PhoneNumbers (locationId, number, description) VALUES (:locationId, :number, :description)", phone_number_params]
    data_set.insert
  end
end
