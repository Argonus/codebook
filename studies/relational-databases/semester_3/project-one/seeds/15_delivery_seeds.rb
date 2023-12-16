# frozen_string_literal: true

class DeliverySeeds
  DELIVERY_STATUS = %w[Pending Accepted Preparing InProgress Delivered]

  def initialize(db)
    @db = db
    @order_type = "Delivery"
    @order_statuses = %w[Pending Accepted Preparing Ready Completed]
    @location_ids = @db["SELECT Id FROM Restaurants.Locations"].to_a.map { |location| location[:id] }
    @address_ids = @db["SELECT Id, CustomerId FROM Clients.Addresses"].to_a.map { |address| { addressId: address[:id], customerId: address[:customerid]} }
    @customer_ids = @db["SELECT Id FROM Clients.Customers"].to_a.map { |customer| customer[:id] }
    @employee_ids = @db["SELECT Id FROM Staff.Employees"].to_a.map { |employee| employee[:id] }
    @dish_ids = @db["SELECT Id FROM Resources.Dishes"].to_a.map { |dish| dish[:id] }
  end

  def run
    create_delivery_status()

    @location_ids.each do |location_id|
      500.times do
        puts "Creating delivery for location #{location_id}"
        customer_id = @customer_ids.sample
        address_id = @address_ids.select { |address| address[:customerId] == customer_id }.sample[:addressId]
        order_id = create_order(location_id, customer_id, address_id)
        delivery_id = create_delivery(order_id)

        rand(1..5).times.map do
          dish_id = @dish_ids.sample
          orderItemId = create_order_item(order_id, dish_id)
          create_delivery_item(delivery_id, orderItemId)
        end
      end
    end
  end

  def create_delivery_status
    DELIVERY_STATUS.each do |delivery_status|
      params = { status: delivery_status }
      @db["INSERT INTO Orders.DeliveryStatuses (Status) VALUES (:status)", params].insert
    end
  end

  def create_order(location_id, customer_id, address_id)
    params = {
      customerId: customer_id,
      locationId: location_id,
      addressId: address_id,
      orderType: @order_type,
      status: @order_statuses.sample,
      requestedDateTime: Faker::Time.between(from: 7.days.ago, to: DateTime.now),
    }

    data_set = @db["INSERT INTO Orders.Orders (CustomerId, LocationId, DeliveryAddressId, OrderType, Status, RequestedDateTime) VALUES (:customerId, :locationId, :addressId, :orderType, :status, :requestedDateTime)", params]
    data_set.insert
  end

  def create_delivery(order_id)
    params = {
      orderId: order_id,
      employeeId: @employee_ids.sample,
      status: DELIVERY_STATUS.sample,
    }

    data_set = @db["INSERT INTO Orders.Deliveries (OrderId, EmployeeId, Status) VALUES (:orderId, :employeeId, :status)", params]
    data_set.insert
  end

  def create_order_item(order_id, dish_id)
    params = {
      orderId: order_id,
      dishId: dish_id,
      quantity: 2,
    }

    data_set = @db["INSERT INTO Orders.OrderItems (OrderId, DishId, Quantity) VALUES (:orderId, :dishId, :quantity)", params]
    data_set.insert
  end

  def create_delivery_item(delivery_id, orderItemId)
    params = {
      deliveryId: delivery_id,
      orderItemId: orderItemId,
      quantity: 2,
    }

    data_set = @db["INSERT INTO Orders.DeliveryItems (DeliveryId, OrderItemId, Quantity) VALUES (:deliveryId, :orderItemId, :quantity)", params]
    data_set.insert
  end
end
