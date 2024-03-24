# frozen_string_literal: true

class OrderSeeds
  ORDER_TYPES = %w[Delivery Pickup DineIn]
  ORDER_STATUSES = %w[Pending Accepted Preparing Ready Completed]

  def initialize(db)
    @db = db
    @customer_ids = @db["SELECT Id FROM Clients.Customers"].to_a.map { |customer| customer[:id] }
    @location_ids = @db["SELECT Id FROM Restaurants.Locations"].to_a.map { |location| location[:id] }
    @dish_ids = @db["SELECT Id FROM Resources.Dishes"].to_a.map { |dish| dish[:id] }
  end

  def run
    create_order_types()
    create_order_statuses()

    2000.times do
      order_id = create_order()
      rand(1..5).times do
        create_order_item(order_id)
      end
    end
  end

  def create_order_types
    ORDER_TYPES.each do |order_type|
      params = { type: order_type }
      @db["INSERT INTO Orders.OrderTypes (Type) VALUES (:type)", params].insert
    end
  end

  def create_order_statuses
    ORDER_STATUSES.each do |order_status|
      params = { status: order_status }
      @db["INSERT INTO Orders.OrderStatuses (Status) VALUES (:status)", params].insert
    end
  end

  def create_order
    customer_id = rand(0..5) == 1 ? @customer_ids.sample : nil
    order_type = ORDER_TYPES.last(2).sample
    order_status = ORDER_STATUSES.sample

    params = {
      customerId: customer_id,
      locationId: @location_ids.sample,
      orderType: order_type,
      status: order_status,
      requestedDateTime: Faker::Time.between(from: 7.days.ago, to: DateTime.now),
    }

    data_set = @db["INSERT INTO Orders.Orders (CustomerId, LocationId, orderType, Status, RequestedDateTime) VALUES (:customerId, :locationId, :orderType, :status, :requestedDateTime)", params]
    data_set.insert
  end

  def create_order_item(order_id)
    params = {
      orderId: order_id,
      dishId: @dish_ids.sample,
      quantity: rand(1..5),
    }

    data_set = @db["INSERT INTO Orders.OrderItems (OrderId, DishId, Quantity) VALUES (:orderId, :dishId, :quantity)", params]
    data_set.insert
  end
end
