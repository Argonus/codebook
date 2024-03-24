# fronze_string_literal: true

class EmployeeLocationRolesSeeds
  def initialize(db)
    @db = db
    @employee_ids = @db["SELECT * FROM Staff.Employees"].to_a.map { |employee| employee[:id] }
    @role_ids = @db["SELECT * FROM Staff.Roles"].to_a.map { |role| role[:id] }
  end

  def run
    @employee_ids.each do |employee_id|
      city_id = @db["SELECT * FROM Staff.Addresses WHERE EmployeeId = ?", employee_id].to_a.sample[:cityid]
      location_id = @db["SELECT * FROM Restaurants.Addresses WHERE CityId = ?", city_id].to_a.sample[:locationid]
      3.times.map { @role_ids.sample }.uniq.each do |role_id|
        create_location_assignment(employee_id, location_id, role_id)
      end
    end
  end

  def create_location_assignment(employee_id, location_id, role_id)
    assignment_params = {
      employeeId: employee_id,
      locationId: location_id,
      roleId: role_id,
      validFrom: Faker::Date.between(from: 1.year.ago, to: Date.today),
      validTo: Faker::Date.between(from: Date.today, to: 1.year.from_now),
    }
    data_set = @db["INSERT INTO Staff.EmployeeLocationRoles (employeeId, locationId, roleId, validFrom, validTo) VALUES (:employeeId, :locationId, :roleId, :validFrom, :validTo)", assignment_params]
    data_set.insert
  end
end
