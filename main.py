import pandas as pd 
import numpy as np
import streamlit as st
from streamlit_folium import folium_static
import folium 

def load_data(): 
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
  def get_flights():
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
   def create_map(): 
      # Creates map object
      self.map = folium.Map(location=[4,-74], tiles="OpenStreetMap", zoom_start=5)
      for city in range(0, self.n_cities):
        folium.Marker(location=[self.cities_airports.iloc[city]['lat'],self. cities_airports.iloc[city]['lng']],popup = "-Ciudad : " + self.cities_airports.iloc[city]['city'] + "\n" + " -Departamento : " + self.cities_airports.iloc[city]['admin_name']  + "\n" + "-Codigo ciudad : " + self.cities_airports.iloc[city]['IATA']).add_to(map)
        lines = folium.PolyLine(lines_points).add_to(self.map)
      return st.markdown(self.map._repr_html_(), unsafe_allow_html=True)
st.set_page_config(
    page_title="Lab 02-Ed2",
    layout="centered",
    initial_sidebar_state="auto",
)
st.title('Flight map')
st.write('Placeholder') 
load_data()
map_data = data(colombia_airports, colombia_cities, colombia_flights)
map_data.create_map()
