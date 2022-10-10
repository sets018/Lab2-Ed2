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
    self.city_list = self.cities_airports["city"].tolist()
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
    self.get_distances(self.cities_airports, self.lines)
  def get_distances(self,cities_airports,lines):
    self.lines_distance = {}
    for line in lines:
      cities = line.split("-")
      for i in range(0, len(cities), 2):
        city_1,city_2 = cities[i],cities[i+1]
        lat_1,lng_1 = float(self.cities_airports[self.cities_airports["IATA"] == city_1]["lat"]),float(self.cities_airports[self.cities_airports["IATA"] == city_1]["lng"])
        lat_2,lng_2 = float(self.cities_airports[self.cities_airports["IATA"] == city_2]["lat"]),float(self.cities_airports[self.cities_airports["IATA"] == city_2]["lng"])
        distance = self.distance(lat_1, lng_1, lat_2, lng_2, 6371)
      self.lines_distance.update({line: distance})
  def distance(self,lat1, lon1, lat2, lon2, r):
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
  def get_nodes_dict(self):
    self.vertices = []
    for i in range(0, len(self.airports_codes)):
      self.vertices.append(i)
    self.nodes_dict = {}
    i = 0
    for city in self.airports_codes:
       self.nodes_dict.update({city: i})
       i = i + 1
    self.get_edges()
  def get_edges(self):
    self.edges = []
    for line in self.lines:
      cities = line.split("-")
      for i in range(0, len(cities), 2):
        city_1,city_2 = cities[i],cities[i+1]
        temp = (self.nodes_dict.get(city_1),self.nodes_dict.get(city_2))
      self.edges.append(temp)
    self.get_distances_coded(self.cities_airports, self.lines)
  def get_distances_coded(self,cities_airports,lines):
    self.lines_distance_coded = {}
    for line in lines:
      cities = line.split("-")
      for i in range(0, len(cities), 2):
        city_1,city_2 = cities[i],cities[i+1]
        lat_1,lng_1 = float(self.cities_airports[self.cities_airports["IATA"] == city_1]["lat"]),float(self.cities_airports[self.cities_airports["IATA"] == city_1]["lng"])
        lat_2,lng_2 = float(self.cities_airports[self.cities_airports["IATA"] == city_2]["lat"]),float(self.cities_airports[self.cities_airports["IATA"] == city_2]["lng"])
        distance = self.distance(lat_1, lng_1, lat_2, lng_2, 6371)
      line_coded = self.nodes_dict.get(city_1),self.nodes_dict.get(city_2)
      self.lines_distance_coded.update({line_coded : distance})
map_data = data(colombia_airports, colombia_cities, colombia_flights)
      
class graph():
  def __init__(self, nodes, edges, distances):
    self.nodes = nodes 
    self.edges = edges 
    self.distances = distances
  def create_path_matrix(self):
    self.path_matrix = [self.nodes]
    for node in self.nodes:
      np.append(self.path_matrix, [self.nodes], axis=0)
  def create_matrix(self):
    matrix = [[0 for i in range(len(self.nodes))] for j in range(len(self.nodes))] 
    for y in range(len(self.nodes)):
      row = []
      for x in range(len(self.nodes)):
        row.append(0)
    matrix.append(row)
    return matrix
  def create_ady_matrix(self):
    self.ady_matrix = self.create_matrix()
    for edge in self.edges:
      x = edge[0]
      y = edge[1]
      self.ady_matrix[x - 1][y - 1] = 1
  def create_dist_matrix(self):
    self.dist_matrix = self.create_matrix()
    for i in range(len(self.nodes)):
      for j in range(len(self.nodes)):
        x = self.nodes[i]
        y = self.nodes[j]
        pos_edge = x,y
        dist = self.distances.get(pos_edge)
        if (x == y):
          self.dist_matrix[x - 1][y - 1] = 0
        else:
          dist = self.distances.get(pos_edge)
          if (dist != None):
            self.dist_matrix[x - 1][y - 1] = dist
          else:
            self.dist_matrix[x - 1][y - 1] = np.inf
  def create_path_matrix(self):
    self.path_matrix = self.create_matrix()
    for x in range(len(self.nodes)):
      for y in range(len(self.nodes)):
        self.path_matrix[x][y] = str(x)
  def floyd(self):
    self.create_path_matrix()
    self.create_ady_matrix()
    self.create_dist_matrix()
    for k in range(len(self.nodes)):
      for i in range(len(self.nodes)):
        for j in range(len(self.nodes)):
          self.dist_matrix[i][j] = min(self.dist_matrix[i][j], self.dist_matrix[i][k] + self.dist_matrix[k][j])
          self.path_matrix[i][j] = self.path_matrix[i][j] + ',' + str(k)
  def print_results(self):
   for x in range(len(self.nodes)):
    for y in range(len(self.nodes)):
      if(self.dist_matrix[x][y] == np.inf):
        print("INF", end=" ")
      else:
        print(self.dist_matrix[x][y] , end="  ")
        print(" ")
st.set_page_config(
    page_title="Lab 02-Ed2",
    layout="centered",
    initial_sidebar_state="auto",
)

class user_input():
    def __init__(self, var, type, data, type_data, input_list):
        self.var = var
        self.type = type
        self.data = data
        self.type_data = type_data
        self.input_list = input_list
        self.get_input()
    def get_input(self):
        self.user_input = 'placeholder'
        if (self.type == 'radio'):
            self.get_radio()
        elif (self.type == 'slider'):
            self.get_slider()
        self.input_list.append(self.user_input)
    def get_radio(self):
        if (self.type_data == 'dataframe'):
            self.user_input = st.radio(
                self.var,
                np.unique(self.data.data_source[self.var]))
        elif (self.type_data == 'list'):
            self.user_input = st.radio(
                self.var,
                self.data)
        st.write(self.var,": ",self.user_input)
    def get_slider(self):
        if (self.type_data == 'dataframe'):
            self.user_input = st.slider(self.var, 0, max(self.data.data_source[self.var]), 1)
        elif (self.type_data == 'list'):
            self.user_input = st.slider(self.var, 0, max(self.data), 1)
        st.write(self.var,": ",self.user_input)
        
st.title('Flight map')
st.write('Placeholder') 
map_data = data(colombia_airports, colombia_cities, colombia_flights)

if st.button('Show map'):
  map_data.create_map()
  
input_columns = ['City origin', 'City destination']
cat_input = []

if st.button('Find shortest distance'):
  city_graph = graph(map_data.vertices,map_data.edges,map_data.lines_distance_coded)
  city_graph.floyd()
with st.sidebar:
  for column in input_columns:
    city_input = user_input(column, 'radio', map_data.city_list, 'list', cat_input)
