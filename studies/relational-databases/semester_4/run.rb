require "sequel"
require "tiny_tds"
C:\Users\argon\Projects\codebook\studies\relational-databases\semester_4
class Solution
  def initialize
    @master = Sequel.connect({
      adapter: "tinytds",
      host: "LAPTOP-TVR18281",
      database: "FoodCourt",
      integrated_security: "SSPI", # Use Integrated Security
      port: 1433, # This is the default SQL Server port
      azure: false, # Set to true if connecting to an Azure SQL Database
      tds_version: "7.3", # Change if using a different TDS version
      appname: "Sequel-TinyTds", # Optional application name for SQL Server logs4
    })
  end

  def run
  end

  private
end
