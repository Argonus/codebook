# frozen_string_literal: true

class NumberTypesSeeds
  def initialize(db)
    @db = db
  end

  def run
    %w[Glowny Awaryjny Kontakt].each do |number_type|
      data_set = @db["INSERT INTO Staff.PhoneNumberTypes (type) VALUES (:type)", { type: number_type }]
      data_set.insert
    end
  end
end
