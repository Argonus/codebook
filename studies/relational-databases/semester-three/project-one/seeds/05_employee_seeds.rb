# frozen_string_literal: true

class EmployeeSeeds
  def initialize(db)
    @db = db
    @location_ids = @db["SELECT * FROM Restaurants.Locations"].to_a.map { |location| location[:id] }
  end

  def run
    70.times do |n|
      employee_id = create_employee()
      location_id = @location_ids.sample
      city_id = @db["SELECT * FROM Restaurants.Addresses WHERE locationId = :id", id: location_id].to_a.first[:cityid]
      create_address(employee_id, city_id)
      create_number(employee_id, "Glowny")
      create_number(employee_id, ["Awaryjny", "Kontakt"].sample)
    end
  end

  def create_employee
    employee_params = {
      firstName: Faker::Name.first_name,
      lastName: Faker::Name.last_name,
      birthDate: Faker::Date.birthday(min_age: 18, max_age: 65),
    }
    data_set = @db["INSERT INTO Staff.Employees (firstName, lastName, birthDate) VALUES (:firstName, :lastName, :birthDate)", employee_params]
    data_set.insert
  end

  def create_address(employee_id, city_id)
    address_params = {
      addressOne: Faker::Address.street_address,
      addressTwo: Faker::Address.secondary_address,
      zipCode: Faker::Address.zip_code[0..4],
      cityId: city_id,
      employeeId: employee_id,
    }
    data_set = @db["INSERT INTO Staff.Addresses (addressOne, addressTwo, zipCode, cityId, employeeId) VALUES (:addressOne, :addressTwo, :zipCode, :cityId, :employeeId)", address_params]
    data_set.insert
  end

  def create_number(employee_id, numberType)
    number_params = {
      employeeId: employee_id,
      numberType: numberType,
      number: Faker::PhoneNumber.cell_phone_in_e164[0..14],
      description: Faker::PhoneNumber.extension,
    }
    data_set = @db["INSERT INTO Staff.PhoneNumbers (employeeId, numberType, number, description) VALUES (:employeeId, :numberType, :number, :description)", number_params]
    data_set.insert
  end
end
