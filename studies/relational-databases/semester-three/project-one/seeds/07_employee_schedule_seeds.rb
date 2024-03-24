# frozen_string_literal: true

class EmployeeScheduleSeeds
  def initialize(db)
    @db = db
    @employee_ids = @db["SELECT * FROM Staff.Employees"].to_a.map { |employee| employee[:id] }
  end

  def run
    @employee_ids.each do |employee_id|
      create_schedule(employee_id)
    end
  end

  def create_schedule(employee_id)
    location_ids = @db["SELECT * FROM Staff.EmployeeLocationRoles WHERE EmployeeId = ?", employee_id].to_a.map { |location| location[:locationid] }
    location_ids.sample(location_ids.count).each do |location_id|
      monday = Faker::Date.between(from: 1.year.ago, to: Date.today).prev_occurring(:monday)
      end_date = Faker::Date.between(from: Date.today, to: 1.year.from_now).next_occurring(:sunday)

      7.times do |dayOfWeek|
        schedule_id = create_schedule_record(monday, end_date, dayOfWeek, employee_id, location_id)
        create_working_hours(schedule_id)
      end
    end
  end

  def create_schedule_record(monday, end_date, dayOfWeek, employee_id, location_id)
    schedule_params = {
      employeeId: employee_id,
      locationId: location_id,
      startDate: monday + dayOfWeek,
      repeatFrequency: [1, 2, 3, 4].sample,
      dayOfWeek: dayOfWeek,
      endDate: end_date + dayOfWeek,
    }
    data_set = @db["INSERT INTO Staff.Schedules (employeeId, locationId, repeatFrequency, startDate, dayOfWeek, endDate) VALUES (:employeeId, :locationId, :repeatFrequency, :startDate, :dayOfWeek, :endDate)", schedule_params]
    data_set.insert
  end

  def create_working_hours(schedule_id)
    working_hours_params = {
      scheduleId: schedule_id,
      startTime: Faker::Time.backward(days: 5, period: :morning, format: :short).to_time,
      endTime: Faker::Time.backward(days: 5, period: :evening, format: :short).to_time,
    }
    data_set = @db["INSERT INTO Staff.WorkingHours (scheduleId, startTime, endTime) VALUES (:scheduleId, :startTime, :endTime)", working_hours_params]
    data_set.insert
  end
end
