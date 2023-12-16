# frozen_string_literal: true

class LocationSchedulesSeeds
  def initialize(db)
    @db = db
    @monday = Date.today.beginning_of_week
  end

  def run
    location_ids = @db["SELECT * FROM Restaurants.Locations"].to_a.map { |location| location[:id] }
    location_ids.each do |location_id|
      7.times do |dayOfWeek|
        scheduleId = create_location_schedule(dayOfWeek, location_id)
      end
    end
  end

  def create_location_schedule(dayOfWeek, location_id)
    location_schedule_params = {
      locationId: location_id,
      dayOfWeek: dayOfWeek,
      startDate: @monday + dayOfWeek,
      endDate: @monday + 30 + dayOfWeek,
    }
    data_set = @db["INSERT INTO Restaurants.Schedules (locationId, dayOfWeek, startDate, endDate) VALUES (:locationId, :dayOfWeek, :startDate, :endDate)", location_schedule_params]
    data_set.insert
  end

  def create_opening_hours(scheduleId)
    opening_hours_params = {
      scheduleId: scheduleId,
      startTime: Time.parse("09:00:00"),
      endTime: Time.parse("17:00:00"),
    }
    data_set = @db["INSERT INTO Restaurants.OpeningHours (scheduleId, startTime, endTime) VALUES (:scheduleId, :startTime, :endTime)", opening_hours_params]
    data_set.insert
  end
end
