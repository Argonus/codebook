# frozen_string_literal: true
require "rgeo/geo_json"
require "csv"

class FileParser
  def initialize(input, output)
    @input = input
    @output = output
  end

  def parse
    file_content = File.read(@input)
    geo_content = RGeo::GeoJSON.decode(file_content)

    CSV.open(@output, "wb") do |csv|
      csv << %w[x y z]
      geo_content.each do |feature|
        line = parse_feature(feature)
        csv << [line[:x], line[:y], line[:z]]
      end
    end
  end

  def parse_feature(feature)
    properties = feature.properties
    x1, y1, x2, y2 = properties["bbox"]

    x = (x1 + x2) / 2
    y = (y1 + y2) / 2
    z = properties["elev"]

    { x: x, y: y, z: z }
  end
end

# https://data.world/city-of-bloomington/457b77d4-0c10-43a0-bd59-aef2923ff26f
input = "#{Dir.pwd}/lib/data.geojson"
output = "#{Dir.pwd}/lib/data.csv"

FileParser.new(input, output).parse