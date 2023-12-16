require "sequel"
require "tiny_tds"
require "faker"
require "date"
require "active_support/all"

require_relative "./_location_seeds.rb"
require_relative "./02_location_schedules_seeds"
require_relative "./03_role_seeds"
require_relative "./04_number_types_seeds"
require_relative "./05_employee_seeds"
require_relative "./06_employee_location_roles_seeds"
require_relative "./07_employee_schedule_seeds"
require_relative "./08_product_seeds"
require_relative "./09_dish_seeds"
require_relative "./10_menu_seeds"
require_relative "./11_customer_seeds"
require_relative "./12_location_table_seeds"
require_relative "./13_table_reservations_seeds"
require_relative "./14_order_seeds"
require_relative "./15_delivery_seeds"
require_relative "./16_dish_product_seeds"

connection_options = {
  adapter: "tinytds",
  host: "LAPTOP-TVR18281",
  database: "FoodCourt",
  integrated_security: "SSPI", # Use Integrated Security
  port: 1433, # This is the default SQL Server port
  azure: false, # Set to true if connecting to an Azure SQL Database
  tds_version: "7.3", # Change if using a different TDS version
  appname: "Sequel-TinyTds", # Optional application name for SQL Server logs4
}

# Establish the connection
DB = Sequel.connect(connection_options)
begin
  DB.transaction do
    DB.run("SET ANSI_NULLS ON")
    DB.run("SET QUOTED_IDENTIFIER ON")
    DB.run("SET CONCAT_NULL_YIELDS_NULL ON")
    DB.run("SET ANSI_WARNINGS ON")
    DB.run("SET ANSI_PADDING ON")

    LocationSeeds.new(DB).run
    puts "Location seeds complete."
    LocationSchedulesSeeds.new(DB).run
    puts "Location schedules seeds complete."
    RoleSeeds.new(DB).run
    puts "Roles seeds complete."
    NumberTypesSeeds.new(DB).run
    puts "Number types seeds complete."
    EmployeeSeeds.new(DB).run
    puts "Employee seeds complete."
    EmployeeLocationRolesSeeds.new(DB).run
    puts "Employee location roles seeds complete."
    EmployeeScheduleSeeds.new(DB).run
    puts "Employee schedules seeds complete."
    ProductSeeds.new(DB).run
    puts "Product seeds complete."
    DishSeeds.new(DB).run
    puts "Dish seeds complete."
    MenuSeeds.new(DB).run
    puts "Menu seeds complete."
    CustomerSeeds.new(DB).run
    puts "Customer seeds complete."
    LocationTableSeeds.new(DB).run
    puts "Location table seeds complete."
    TableReservationsSeeds.new(DB).run
    puts "Table reservations seeds complete."
    OrderSeeds.new(DB).run
    puts "Order seeds complete."
    DeliverySeeds.new(DB).run
    puts "Delivery seeds complete."
    DishProductsSeeds.new(DB).run
    puts "Dish products seeds complete."
  end
ensure
  DB.disconnect
end
