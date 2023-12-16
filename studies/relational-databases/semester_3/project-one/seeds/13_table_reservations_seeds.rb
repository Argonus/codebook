# frozen_string_literal: true

class TableReservationsSeeds
  def initialize(db)
    @db = db
    @location_ids = @db["SELECT Id FROM Restaurants.Locations"].map { |row| row[:id] }
    @customer_ids = @db["SELECT Id FROM Clients.Customers"].map { |row| row[:id] }
  end

  def run
    @location_ids.each do |location_id|
      table_ids = @db["SELECT Id FROM Restaurants.Tables WHERE LocationId = :locationId", locationId: location_id].map { |row| row[:id] }
      (0..6).each do |day|
        next if rand(2) == 0
        reservation_id = create_reservation(day)
        table_ids.sample(rand(1..table_ids.length)).each do |table_id|
          create_table_reservation(table_id, reservation_id)
        end
      end
    end
  end

  def create_reservation(day)
    reservation_params = {
      customerId: @customer_ids.sample,
      seats: rand(1..4),
      notes: Faker::Lorem.sentence[0..49],
      reservationDate: Date.today + day,
      reservationHour: Time.now,
      reservationTime: rand(3600..7200),
    }
    data_set = @db["INSERT INTO Restaurants.Reservations (CustomerId, Seats, Notes, ReservationDate, ReservationHour, ReservationTime) VALUES (:customerId, :seats, :notes, :reservationDate, :reservationHour, :reservationTime)", reservation_params]
    data_set.insert
  end

  def create_table_reservation(table_id, reservation_id)
    params = {
      tableId: table_id,
      reservationId: reservation_id,
    }
    data_set = @db["INSERT INTO Restaurants.TableReservations (TableId, ReservationId) VALUES (:tableId, :reservationId)", params]
    data_set.insert
  end
end
