import pandas as pd 
import numpy as np
from streamlit_folium import st_folium
import streamlit as st
import folium 
from math import radians, cos, sin, asin, acos, sqrt, pi
from geopy import distance
from geopy.geocoders import Nominatim
import osmnx as ox
import networkx as nx
colombia_airports_url = "https://raw.githubusercontent.com/sets018/Lab2-Ed2/main/data/colombia_airports.csv"
colombia_airports = pd.read_csv(colombia_airports_url)
colombia_cities_url = "https://raw.githubusercontent.com/sets018/Lab2-Ed2/main/data/colombia_cities.csv"
colombia_cities = pd.read_csv(colombia_cities_url)
colombia_cities = colombia_cities[colombia_cities["country"] == "Colombia"]
colombia_flights_url = 'https://raw.githubusercontent.com/sets018/Lab2-Ed2/main/data/colombia_flights.csv'
colombia_flights = pd.read_csv(colombia_flights_url)
colombia_flights = colombia_flights[(colombia_flights["Pais Origen"] == "COLOMBIA") & (colombia_flights["Pais Destino"] == "COLOMBIA")]
  
class data():
  def __init__(self, colombia_airports, colombia_cities, colombia_flights):
    self.colombia_airports = colombia_airports
    self.colombia_cities = colombia_cities
    self.colombia_flights = colombia_flights
    self.join_data()
  def join_data(self):
    self.colombia_cities_airports = pd.merge(colombia_cities, colombia_airports, how='inner', left_on = 'city', right_on = 'City served')
    self.cities_airports = self.colombia_cities_airports[["city","lat","lng","admin_name","id","Airport Name","ICAO","IATA","Category"]].copy()
    self.n_cities = self.cities_airports.shape[0]
    self.get_airports_codes()
  def get_airports_codes(self):
     self.airports_codes = []
     self.codes = self.cities_airports.IATA.unique()
     for code in self.codes:
      self.airports_codes.append(code)
     self.get_flights()
  def get_flights(self):
    self.colombia_flights = self.colombia_flights[(self.colombia_flights["Origen"].isin(self.airports_codes)) & (self.colombia_flights["Destino"].isin(self.airports_codes))]
    self.colombia_flights_real = self.colombia_flights[self.colombia_flights["Tipo Vuelo"] == 'R']
    self.colombia_flights_real['lines'] = self.colombia_flights_real["Origen"] + '-' + self.colombia_flights_real["Destino"]
    self.lines = self.colombia_flights_real["lines"].unique()
    self.lines_points = []
    for line in self.lines:
      self.cities = line.split("-")
      for city_code in self. cities: 
        city_coords = (float(self.cities_airports[self.cities_airports["IATA"] == city_code]["lat"]),float(self.cities_airports[self.cities_airports["IATA"] == city_code]["lng"]))
        self.lines_points.append(city_coords)
  def get_distances(self,cities_airports,lines):
    cities_airports = self.cities_airports
    lines = self.lines 
    lines_distance = {}
    for line in lines:
      cities = line.split("-")
      for i in range(0, len(cities), 2):
        city_1,city_2 = cities[i],cities[i+1]
        lat_1,lng_1 = float(data.cities_airports[data.cities_airports["IATA"] == city_1]["lat"]),float(data.cities_airports[data.cities_airports["IATA"] == city_1]["lng"])
        lat_2,lng_2 = float(data.cities_airports[data.cities_airports["IATA"] == city_2]["lat"]),float(data.cities_airports[data.cities_airports["IATA"] == city_2]["lng"])
        distance = distance(lat_1, lng_1, lat_2, lng_2, r=6371)
      lines_distance.update({line: distance})
  def calculate_spherical_distance(lat1, lon1, lat2, lon2, r=6371):
    #Convert degrees to radians
    coordinates = lat1, lon1, lat2, lon2
    #radians(c) is same as c*pi/180
    phi1, lambda1, phi2, lambda2 = [
        radians(c) for c in coordinates
    ] 
    # Apply the haversine formula
    a = (np.square(sin((phi2-phi1)/2)) + cos(phi1) * cos(phi2) * 
        np.square(sin((lambda2-lambda1)/2)))
    d = 2*r*asin(np.sqrt(a))
    return d
  def create_map(self): 
    # Creates map object
    map = folium.Map(location=[4,-74], tiles="OpenStreetMap", zoom_start=5)
    for city in range(0, self.n_cities):
      folium.Marker(location=[self.cities_airports.iloc[city]['lat'],self. cities_airports.iloc[city]['lng']],popup = "-Ciudad : " + self.cities_airports.iloc[city]['city'] + "\n" + " -Departamento : " + self.cities_airports.iloc[city]['admin_name']  + "\n" + "-Codigo ciudad : " + self.cities_airports.iloc[city]['IATA']).add_to(map)
      lines = folium.PolyLine(self.lines_points).add_to(map)
    map_fig = st_folium(map, key="fig1", width=700, height=700)
st.set_page_config(
    page_title="Lab 02-Ed2",
    layout="centered",
    initial_sidebar_state="auto",
)
st.title('Flight map')
st.write('Placeholder') 
map_data = data(colombia_airports, colombia_cities, colombia_flights)
map_data.create_map()
