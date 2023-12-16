# frozen_string_literal: true

class RoleSeeds
  def initialize(db)
    @db = db
  end

  def run
    ["General Manager", "Executive Chef", "Souse Chef", "Cook", "Waiter", "Delivery Driver"].map do |role|
      role_id = create_role(role)
      { id: role_id, name: role }
    end
  end

  def create_role(role)
    role_params = { name: role }
    data_set = @db["INSERT INTO Staff.Roles (Name) VALUES (:name)", role_params]
    data_set.insert
  end
end
