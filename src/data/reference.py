"""Static reference data used by the data-collection simulator.

Airport coordinates, time zones, runway counts and hub status are real; the
operational parameters (capacity, climate zone) are calibrated approximations
used to drive the simulation.
"""
from __future__ import annotations

# code, name, city, state, region, lat, lon, tz, runways, elevation_ft,
# hub(1/0), hourly_capacity (departures the field can absorb per hour),
# climate zone
AIRPORTS: list[tuple] = [
    ("ATL", "Hartsfield-Jackson Atlanta Intl", "Atlanta", "GA", "Southeast", 33.6407, -84.4277, "America/New_York", 5, 1026, 1, 105, "humid_subtropical"),
    ("DFW", "Dallas/Fort Worth Intl", "Dallas", "TX", "South", 32.8998, -97.0403, "America/Chicago", 7, 607, 1, 95, "humid_subtropical"),
    ("DEN", "Denver Intl", "Denver", "CO", "Mountain", 39.8561, -104.6737, "America/Denver", 6, 5434, 1, 90, "semi_arid_cold"),
    ("ORD", "Chicago O'Hare Intl", "Chicago", "IL", "Midwest", 41.9742, -87.9073, "America/Chicago", 8, 672, 1, 88, "continental"),
    ("LAX", "Los Angeles Intl", "Los Angeles", "CA", "West", 33.9416, -118.4085, "America/Los_Angeles", 4, 125, 1, 78, "mediterranean"),
    ("CLT", "Charlotte Douglas Intl", "Charlotte", "NC", "Southeast", 35.2144, -80.9473, "America/New_York", 4, 748, 1, 72, "humid_subtropical"),
    ("LAS", "Harry Reid Intl", "Las Vegas", "NV", "West", 36.0840, -115.1537, "America/Los_Angeles", 4, 2181, 0, 62, "desert"),
    ("PHX", "Phoenix Sky Harbor Intl", "Phoenix", "AZ", "West", 33.4352, -112.0101, "America/Phoenix", 3, 1135, 1, 60, "desert"),
    ("MCO", "Orlando Intl", "Orlando", "FL", "Southeast", 28.4312, -81.3081, "America/New_York", 4, 96, 0, 58, "humid_subtropical"),
    ("SEA", "Seattle-Tacoma Intl", "Seattle", "WA", "Northwest", 47.4502, -122.3088, "America/Los_Angeles", 3, 433, 1, 55, "marine_west"),
    ("MIA", "Miami Intl", "Miami", "FL", "Southeast", 25.7959, -80.2870, "America/New_York", 4, 8, 1, 56, "tropical"),
    ("IAH", "George Bush Intercontinental", "Houston", "TX", "South", 29.9902, -95.3368, "America/Chicago", 5, 97, 1, 62, "humid_subtropical"),
    ("JFK", "John F. Kennedy Intl", "New York", "NY", "Northeast", 40.6413, -73.7781, "America/New_York", 4, 13, 1, 52, "continental"),
    ("EWR", "Newark Liberty Intl", "Newark", "NJ", "Northeast", 40.6895, -74.1745, "America/New_York", 3, 18, 1, 46, "continental"),
    ("SFO", "San Francisco Intl", "San Francisco", "CA", "West", 37.6213, -122.3790, "America/Los_Angeles", 4, 13, 1, 44, "mediterranean"),
    ("MSP", "Minneapolis-St Paul Intl", "Minneapolis", "MN", "Midwest", 44.8848, -93.2223, "America/Chicago", 4, 841, 1, 56, "continental_cold"),
    ("DTW", "Detroit Metro Wayne County", "Detroit", "MI", "Midwest", 42.2162, -83.3554, "America/New_York", 6, 645, 1, 54, "continental_cold"),
    ("BOS", "Boston Logan Intl", "Boston", "MA", "Northeast", 42.3656, -71.0096, "America/New_York", 6, 20, 1, 50, "continental"),
    ("SLC", "Salt Lake City Intl", "Salt Lake City", "UT", "Mountain", 40.7899, -111.9791, "America/Denver", 4, 4227, 1, 48, "semi_arid_cold"),
    ("PHL", "Philadelphia Intl", "Philadelphia", "PA", "Northeast", 39.8744, -75.2424, "America/New_York", 4, 36, 1, 46, "continental"),
    ("FLL", "Fort Lauderdale-Hollywood Intl", "Fort Lauderdale", "FL", "Southeast", 26.0742, -80.1506, "America/New_York", 2, 65, 0, 40, "tropical"),
    ("BWI", "Baltimore/Washington Intl", "Baltimore", "MD", "Northeast", 39.1774, -76.6684, "America/New_York", 3, 146, 0, 40, "continental"),
    ("SAN", "San Diego Intl", "San Diego", "CA", "West", 32.7338, -117.1933, "America/Los_Angeles", 1, 17, 0, 30, "mediterranean"),
    ("LGA", "LaGuardia", "New York", "NY", "Northeast", 40.7769, -73.8740, "America/New_York", 2, 21, 0, 34, "continental"),
    ("IAD", "Washington Dulles Intl", "Washington", "DC", "Northeast", 38.9531, -77.4565, "America/New_York", 4, 313, 0, 42, "continental"),
    ("TPA", "Tampa Intl", "Tampa", "FL", "Southeast", 27.9755, -82.5332, "America/New_York", 3, 26, 0, 34, "humid_subtropical"),
    ("MDW", "Chicago Midway Intl", "Chicago", "IL", "Midwest", 41.7868, -87.7522, "America/Chicago", 5, 620, 0, 34, "continental"),
    ("PDX", "Portland Intl", "Portland", "OR", "Northwest", 45.5898, -122.5951, "America/Los_Angeles", 3, 31, 0, 30, "marine_west"),
    ("DAL", "Dallas Love Field", "Dallas", "TX", "South", 32.8471, -96.8518, "America/Chicago", 5, 487, 0, 28, "humid_subtropical"),
    ("STL", "St. Louis Lambert Intl", "St. Louis", "MO", "Midwest", 38.7487, -90.3700, "America/Chicago", 4, 618, 0, 28, "continental"),
    ("AUS", "Austin-Bergstrom Intl", "Austin", "TX", "South", 30.1975, -97.6664, "America/Chicago", 2, 542, 0, 26, "humid_subtropical"),
    ("RDU", "Raleigh-Durham Intl", "Raleigh", "NC", "Southeast", 35.8801, -78.7880, "America/New_York", 3, 435, 0, 24, "humid_subtropical"),
    ("BNA", "Nashville Intl", "Nashville", "TN", "Southeast", 36.1263, -86.6774, "America/Chicago", 4, 599, 0, 28, "humid_subtropical"),
    ("MSY", "Louis Armstrong New Orleans Intl", "New Orleans", "LA", "South", 29.9934, -90.2580, "America/Chicago", 2, 4, 0, 22, "humid_subtropical"),
    ("DCA", "Ronald Reagan Washington National", "Washington", "DC", "Northeast", 38.8512, -77.0402, "America/New_York", 3, 15, 0, 32, "continental"),
]

AIRPORT_COLUMNS = [
    "airport_code", "airport_name", "city", "state", "region", "latitude",
    "longitude", "timezone", "num_runways", "elevation_ft", "is_hub",
    "hourly_capacity", "climate_zone",
]

# carrier, name, type, fleet_share, ops_quality (higher = worse on-time),
# primary hubs
AIRLINES: list[tuple] = [
    ("AA", "American Airlines", "legacy", 0.135, 0.10, ["DFW", "CLT", "PHX", "MIA", "ORD"]),
    ("DL", "Delta Air Lines", "legacy", 0.135, -0.22, ["ATL", "DTW", "MSP", "SLC", "LAX"]),
    ("UA", "United Airlines", "legacy", 0.120, 0.05, ["ORD", "EWR", "IAH", "DEN", "SFO"]),
    ("WN", "Southwest Airlines", "low_cost", 0.155, 0.02, ["MDW", "DAL", "LAS", "BWI", "PHX"]),
    ("AS", "Alaska Airlines", "low_cost", 0.055, -0.15, ["SEA", "PDX", "SAN"]),
    ("B6", "JetBlue Airways", "low_cost", 0.065, 0.32, ["JFK", "BOS", "FLL", "MCO"]),
    ("NK", "Spirit Airlines", "ultra_low_cost", 0.060, 0.28, ["FLL", "MCO", "LAS", "DTW"]),
    ("F9", "Frontier Airlines", "ultra_low_cost", 0.045, 0.35, ["DEN", "MCO", "LAS"]),
    ("G4", "Allegiant Air", "ultra_low_cost", 0.030, 0.22, ["LAS", "TPA", "MCO"]),
    ("OO", "SkyWest Airlines", "regional", 0.085, 0.12, ["SLC", "DEN", "ORD", "LAX"]),
    ("YX", "Republic Airways", "regional", 0.060, 0.18, ["DCA", "LGA", "PHL", "ORD"]),
    ("MQ", "Envoy Air", "regional", 0.055, 0.20, ["DFW", "ORD", "MIA", "LGA"]),
]

AIRLINE_COLUMNS = ["carrier_code", "carrier_name", "carrier_type", "fleet_share",
                   "ops_quality_index", "hubs"]

# model, manufacturer, seats, range_km, category, reliability (lower = better)
AIRCRAFT_TYPES: list[tuple] = [
    ("B737-800", "Boeing", 175, 5400, "narrowbody", 0.00),
    ("B737-900ER", "Boeing", 189, 5900, "narrowbody", 0.01),
    ("B737 MAX 8", "Boeing", 178, 6500, "narrowbody", -0.04),
    ("A320-200", "Airbus", 168, 5700, "narrowbody", 0.02),
    ("A321-200", "Airbus", 190, 5900, "narrowbody", 0.01),
    ("A321neo", "Airbus", 196, 7400, "narrowbody", -0.05),
    ("A319-100", "Airbus", 128, 6900, "narrowbody", 0.03),
    ("B757-200", "Boeing", 199, 7200, "narrowbody", 0.09),
    ("E175", "Embraer", 76, 3900, "regional_jet", 0.01),
    ("E145", "Embraer", 50, 2800, "regional_jet", 0.10),
    ("CRJ-900", "Bombardier", 76, 2900, "regional_jet", 0.06),
    ("CRJ-200", "Bombardier", 50, 3100, "regional_jet", 0.12),
    ("B767-300", "Boeing", 261, 11000, "widebody", 0.08),
    ("A220-300", "Airbus", 137, 6300, "narrowbody", -0.06),
]

AIRCRAFT_TYPE_COLUMNS = ["aircraft_model", "manufacturer", "seats", "range_km",
                         "category", "reliability_index"]

# Fleet composition by carrier type
FLEET_BY_CARRIER_TYPE = {
    "legacy": ["B737-800", "B737-900ER", "A320-200", "A321-200", "A319-100",
               "B757-200", "B767-300", "A321neo"],
    "low_cost": ["B737-800", "B737 MAX 8", "A320-200", "A321neo", "A220-300",
                 "E175"],
    "ultra_low_cost": ["A320-200", "A321-200", "A319-100", "A321neo"],
    "regional": ["E175", "E145", "CRJ-900", "CRJ-200"],
}

CLIMATE_PARAMS = {
    # zone: (mean_temp_c, seasonal_amp, wet_day_prob, mean_precip_mm,
    #        snow_capable, fog_prob, storm_season_peak_month)
    "humid_subtropical": (18.5, 10.0, 0.28, 3.2, 0, 0.05, 7),
    "continental":       (12.0, 13.0, 0.30, 2.6, 1, 0.06, 7),
    "continental_cold":  (8.5, 15.5, 0.31, 2.3, 1, 0.07, 6),
    "semi_arid_cold":    (10.5, 13.0, 0.20, 1.5, 1, 0.03, 7),
    "desert":            (24.0, 11.0, 0.07, 1.1, 0, 0.01, 8),
    "mediterranean":     (16.5, 5.5, 0.16, 2.0, 0, 0.12, 1),
    "marine_west":       (11.5, 7.5, 0.42, 2.4, 0, 0.14, 11),
    "tropical":          (25.5, 5.0, 0.34, 4.6, 0, 0.03, 8),
    "south":             (20.0, 10.0, 0.28, 3.0, 0, 0.05, 7),
}
