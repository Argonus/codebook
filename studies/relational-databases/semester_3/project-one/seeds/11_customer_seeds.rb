# frozen_string_literal: true

class CustomerSeeds
  def initialize(db)
    @db = db
    @city_ids = @db["SELECT * FROM GlobalConfig.Cities"].to_a.map { |city| city[:id] }
  end

  def run
    @city_ids.each do |city_id|
      100.times do
        customer_id = create_customer()
        create_customer_address(customer_id, city_id)
      end
    end
  end

  def create_customer
    customer_params = {
      firstName: Faker::Name.first_name[0..49],
      lastName: Faker::Name.last_name[0..49],
      email: Faker::Internet.email[0..99],
      phoneNumber: Faker::PhoneNumber.phone_number[0..14],
      blocked: [0, 1].sample,
    }
    data_set = @db["INSERT INTO Clients.Customers (firstName, lastName, email, phoneNumber) VALUES (:firstName, :lastName, :email, :phoneNumber)", customer_params]
    data_set.insert
  end

  def create_customer_address(customer_id, city_id)
    addressParams = {
      addressOne: Faker::Address.street_address[0..99],
      addressTwo: Faker::Address.secondary_address[0..99],
      zipCode: Faker::Address.zip_code[0..4],
      description: Faker::Address.community[0..99],
      customerId: customer_id,
      cityId: city_id,
    }
    data_set = @db["INSERT INTO Clients.Addresses (addressOne, addressTwo, zipCode, description, customerId, cityId) VALUES (:addressOne, :addressTwo, :zipCode, :description, :customerId, :cityId)", addressParams]
    data_set.insert
  end
end
