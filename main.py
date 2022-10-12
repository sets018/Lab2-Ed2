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
colombia_flights_url = 'https://raw.githubusercontent.com/sets018/Lab2-Ed2/main/data/colombia_flights2.csv'
colombia_flights = pd.read_csv(colombia_flights_url)
colombia_flights = colombia_flights[(colombia_flights["Pais Origen"] == "COLOMBIA") & (colombia_flights["Pais Destino"] == "COLOMBIA")]
 
# Clase que almacena los datos de los vuelos en colombia y los metodos para extraer/procesar los datos
class data():
  def __init__(self, colombia_airports, colombia_cities, colombia_flights):
    # La clase recibe como atributo los dataframes que obtienen los datos extraidos de la pagina de la aerocivil
    self.colombia_airports = colombia_airports
    self.colombia_cities = colombia_cities
    self.colombia_flights = colombia_flights
    self.join_data()
    self.get_nodes_dict()
    self.map_created = 0
  # Funcion que une los datos de las ciudades y los aeropuertos 
  def join_data(self):
    self.colombia_cities_airports = pd.merge(colombia_cities, colombia_airports, how='inner', left_on = 'city', right_on = 'City served')
    self.cities_airports = self.colombia_cities_airports[["city","lat","lng","admin_name","id","Airport Name","ICAO","IATA","Category"]].copy()
    self.city_list = self.cities_airports["city"].tolist()
    self.n_cities = self.cities_airports.shape[0]
    self.get_airports_codes()
  # Funcion que extrae los codigos de los aeropuertos y los almacena en una lista
  def get_airports_codes(self):
     self.airports_codes = []
     self.codes = self.cities_airports.IATA.unique()
     for code in self.codes:
      self.airports_codes.append(code)
     self.cities_name = []
     self.names = self.cities_airports.city.unique()
     for name in self.names:
      self.cities_name.append(name)
     self.get_flights()
  # Funcion que en base a los datos obtiene las lineas de vuelo comerciales en colombia y las ciudades a las que conecta asi como sus coordenadas
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
  # Funcion que calcula las distancias entre las ciudades que estan conectadas lineas de vuelo directas y las almacena en un diccionario donde el key es la ruta
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
  # Funcion que obtiene la distancia entre 2 puntos dadas sus coordenadas ( latitud y longitud ) y devuelve deste dato como un float
  def distance(self,lat1, lon1, lat2, lon2, r):
    coordinates = lat1, lon1, lat2, lon2
    #radians(c) is same as c*pi/180
    phi1, lambda1, phi2, lambda2 = [
        radians(c) for c in coordinates
    ] 
    # Aplica la formula para encontrar distancia entre 2 puntos
    a = (np.square(sin((phi2-phi1)/2)) + cos(phi1) * cos(phi2) * 
        np.square(sin((lambda2-lambda1)/2)))
    d = 2*r*asin(np.sqrt(a))
    return d
  # Funcion que crea el mapa en basse a las lineas y las ciudades capitales de colombia que cuentan con aeropuertos
  def create_map(self): 
    if (self.map_created == 0):
      # Creates map object
      map = folium.Map(location=[4,-74], tiles="OpenStreetMap", zoom_start=5)
      for city in range(0, self.n_cities):
        folium.Marker(location=[self.cities_airports.iloc[city]['lat'],self. cities_airports.iloc[city]['lng']],popup = "-Ciudad : " + self.cities_airports.iloc[city]['city'] + "\n" + " -Departamento : " + self.cities_airports.iloc[city]['admin_name']  + "\n" + "-Codigo ciudad : " + self.cities_airports.iloc[city]['IATA']).add_to(map)
        lines = folium.PolyLine(self.lines_points).add_to(map)
      map_fig = st_folium(map, key="fig1", width=700, height=700)
  # Funcion que genera un diccionario en el que cada ciudad tiene asignada un valor numerico del 0 al 31
  def get_nodes_dict(self):
    self.vertices = []
    for i in range(0, len(self.airports_codes)):
      self.vertices.append(i)
    self.nodes_dict = {}
    i = 0
    for city in self.airports_codes:
       self.nodes_dict.update({city: i})
       i = i + 1
    self.names_dict = {}
    i = 0
    for code in self.airports_codes:
     city = self.names[i]
     self.names_dict.update({city: code})
     i = i + 1
    self.get_edges()
    self.inv_nodes_dict = {i: j for j, i in self.nodes_dict.items()}
  # Funcion que genera una lista con todas las rutas comerciales que conectan 2 ciudades capitales con aeropuerto
  def get_edges(self):
    self.edges = []
    for line in self.lines:
      cities = line.split("-")
      for i in range(0, len(cities), 2):
        city_1,city_2 = cities[i],cities[i+1]
        temp = (self.nodes_dict.get(city_1),self.nodes_dict.get(city_2))
      self.edges.append(temp)
    self.get_distances_coded(self.cities_airports, self.lines)
  # Funcion que calcula las distancias entre las ciudades que estan conectadas lineas de vuelo directas y las almacena en un diccionario donde el key es la ruta con las ciudades en valores numericos
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
      
# Clase para representar el grafo en matrices de adyacencia y de distancia y realizar operaciones (algoritmo de floyd y prim)
# Tambien cuenta con los metodos para extraer los resutados de estas operaciones 
class graph():
  # Recibe los vertices del grafo ( las 32 ciudades capitales con aeropuerto ) almacenados en una lista que son el atributo nodes de la clase grafo 
  # Y las aristas de este ( las 152 rutas comerciales entre ciudades capitales con aeropuerto ) almacenados en una lista de tuplas que son el atributo edges de la clase grafo 
  # Asi como las distancias entre las ciudades conectadas por rutas comerciales almacenados en un diccionario que son el atributo distances de la clase data 
  def __init__(self, nodes, edges, distances, nodes_dict, names_dict):
    self.nodes = nodes 
    self.edges = edges 
    self.distances = distances
    self.create_ady_matrix()
    self.create_dist_matrix()
    self.create_path_matrix()
    self.nodes_dict = nodes_dict
    self.names_dict = names_dict
  # Crea una matriz de 32*32 (numero de vertices x numero de vertices) llena con ceros
  def create_matrix(self):
    matrix = [[0 for i in range(len(self.nodes))] for j in range(len(self.nodes))] 
    return matrix
  # Crea la matriz de adyacencia
  def create_ady_matrix(self):
    # Crea e inicializa la matriz de adyacencia como una matriz llena de ceros 
    self.ady_matrix = self.create_matrix().copy()
    # Itera por todas las aristas 
    for edge in self.edges:
      # Extrae los vertices que hacen parte de cada arista (x,y) 
      x = edge[0]
      y = edge[1]
      # La matriz en la fila x, columna y toma un valor 1 para representar que hay adyacencia entre x y y, es decir que existe una arista que va de x a y
      self.ady_matrix[x][y] = 1
      # En las combinaciones de fila y columnas donde no hay adyacencia la matriz queda con el valor 0 que tenia inicialmente
  # Crea la matriz de distancia
  def create_dist_matrix(self):
    # Crea e inicializa la matriz de distancia como una matriz llena de ceros 
    self.dist_matrix = self.create_matrix().copy()
    # Itera por todas las posibles aristas
    for x in range(len(self.nodes)):
      for y in range(len(self.nodes)):
        # Almacena los vertices que forman una posible arista en una tupla
        pos_edge = (x,y)
        # Si hay una arista bucle su valor es automaticamente 0 ya que la distancia de un  vertice a si mismo es 0
        if (x == y):
          self.dist_matrix[x][y] = 0
        else:
          # Si hay una arista que conecta 2 vertices diferentes se extrae su distancia del diccionario que almacena todas las distancia entre vertices adyacentes 
          # Usando la tupla que almacena los vertices de la arista como key
          dist = self.distances.get(pos_edge)
          # Si la distancia es difernente de None ( la distancia es None cuando la key no existe en el diccionario por lo tanto no hay adyacencia entre los vertices )
          if (dist != None):
              # Entonces la fila y columna relacionadas a los vertices toma el valor de la distancia entre estos obtenida en el diccionario
              self.dist_matrix[x][y] = dist
          # Si la distancia es None 
          else:
              # Entonces la fila y columna relacionadas a los vertices toma el valor de infinito ya que no hay adyacencia entre estos ( No hay arista que una a estos vertices, la distancia entre estos es inifinito)
              self.dist_matrix[x][y] = float('inf')
  # Crea la matriz de recorrido
  def create_path_matrix(self):
    # Crea e inicializa la matriz de distancia como una matriz llena de ceros 
    self.path_matrix = self.create_matrix().copy()
    # Itera por todas las posibles aristas
    for x in range(len(self.nodes)):
      for y in range(len(self.nodes)):
        # Si hay una arista bucle entonces el camino a este se determina como 0
        if (x == y):
          self.path_matrix[x][y] = 0
        # Si hay adyacencia entre este par de vertices entonces se inicializa el camino empezando en x
        elif (self.dist_matrix[x][y] != float('inf')):
          self.path_matrix[x][y] = x
        else:
        # Si no hay adyacencia entre este par de vertices entonces el camino a este se determina como -1
          self.path_matrix[x][y] = -1
  # Ejecuta el algoritmo de floyd 
  def floyd(self, dist_matrix, path_matrix):
    # Se ejecuta el algoritmo de floyd con las matrices de distancia y recorrido
      costs = dist_matrix.copy()
      for k in range(len(self.nodes)):
        for i in range(len(self.nodes)):
          for j in range(len(self.nodes)):
            # Si encuentra un camino minimo actualiza la matriz de caminos y distancia
            if ((costs[i][k] != float('inf')) and (costs[k][j] != float('inf')) and (costs[i][k] + costs[k][j] < costs[i][j])):
                    costs[i][j] = costs[i][k] + costs[k][j]
                    path_matrix[i][j] = path_matrix[k][j]
      # Funcion para extraer el camino de menor costo que va de cualquier vertice a hacia cualquier vertice b 
      self.get_paths(path_matrix)
  # Funcion para encontrar los caminos de menor costo de cualquier vertice a cualquier otro
  def get_paths(self,path_matrix):
    # Diccionario que almacena los caminos minimos posibles entre todos los vertices
    self.paths_dict = {}
    # Itera por todas las posibles aristas ( pares de vertices en el grafo )
    for a in range(len(self.nodes)):
      for b in range(len(self.nodes)):
        # Si los pares de vertices no forman una arista bucle y tienen adyacencia en el grafo entonces se forma la ruta empezando por i 
        if (a != b) and (path_matrix[a][b] != -1):
          # Se inicializa el camino empezando por el punto inicial
          route = [a]
          # Se llama la funcion recursiva que construe el recorrido basado en la matriz de recorrido ya creada
          self.fill_route(path_matrix, a, b, route)
          # Finalmente cuando ya esta construido el camino se termina agregando el punto final
          route.append(b)
          #print(f'The shortest path from {a} —> {b} is', route)
          # Agrega la ruta de costo minimo con el par de vertices en el diccionario junto a su distancia 
          orig = a
          dest = b
          pair = (orig,dest)
          self.paths_dict.update({pair: route})
  # Funcion para obtener la ruta de menor costo que va de cada vertica a a cada vertice b en base a los resultados de aplicar el algoritmo de floyd
  def fill_route(self,path_matrix, a, b, route):
      # Criterio para parar la recursion
      if (path_matrix[a][b] == a):
          return
      # Llama a la funcion recursivamente con un cambio de paramteros
      self.fill_route(path_matrix, a, path_matrix[a][b], route)
      route.append(path_matrix[a][b])
  # Obtiene el camino minimo dados dos vertices especificos ( se pasan los nombres de las ciudades como parametro ) del diccionario con todos los caminos para todos los pares de vertices posibles 
  def extract_usr_path(self,a,b):
    # Extrae los codigos de los aeropuerto de las ciudades del datframe de aeropuertos y ciudades 
    usr_input_a = self.names_dict.get(a)
    usr_input_b = self.names_dict.get(b)
    # Sie l usuario selecciona un punto de partida igual al de entrada entonces es un bucle 
    if (usr_input_a == usr_input_b):
     st.write("Como la ciudad de origen es al misma que la ciudad desstino entonces no hay camino mas corto")
     st.write(usr_input_a, ' -> ',usr_input_b)
    # Extrae los codigos de los aeropuerto del diccionario ( 0 - 32 )
    else:
     usr_a = self.nodes_dict.get(usr_input_a)
     usr_b = self.nodes_dict.get(usr_input_b)
     usr_pair = (usr_a,usr_b)
     # Basado en el punto de origen y de destinacion que da el usuario encuentra la ruta de camino minimo en el diccionario
     self.usr_path = self.paths_dict.get(usr_pair)
     cities = []
     st.write(self.usr_path)
     st.write(list(self.paths_dict.keys())[0],list(self.paths_dict.keys())[18],list(self.paths_dict.keys())[5],list(self.paths_dict.keys())[4],list(self.paths_dict.keys())[1])
     st.write(usr_pair)
     for node in self.usr_path:
      city = self.nodes_dict.get(node)
      cities.append(city)
     return cities
    
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

if st.checkbox('Show map'):
  map_data.create_map()
  map_data.map_created = 1
else: 
  map_data.map_created = 1
  
input_columns = ['City origin (A)', 'City destination (B)']

cat_input = []
cat_input2 = []
  
with st.sidebar:
  if st.checkbox('Find the shortest path beetwen two cities'):
   for column in input_columns:
    city_input = user_input(column, 'radio', map_data.city_list, 'list', cat_input)
   if st.button('Find shortest path from (A) to (B)'):
    cities_graph = graph(map_data.vertices,map_data.edges,map_data.lines_distance_coded,map_data.nodes_dict,map_data.names_dict)
    cities_graph.floyd(cities_graph.dist_matrix,cities_graph.path_matrix)
    usr_input_a = cat_input[0]
    usr_input_b = cat_input[1]
    #cities = cities_graph.test(usr_input_a,usr_input_b)
    cities = cities_graph.extract_usr_path(usr_input_a,usr_input_b)
    if (cities != None):
     for city in cities:
      st.write(cat_input[0],cat_input[1])
      st.write(city, ' -> ')
  if st.checkbox('Find the shortest path to traverse all cities from an origin point'):
    city_input2 = user_input('City origin (A)', 'radio', map_data.city_list, 'list', cat_input2)
    if st.button('Find shortest path from (A) to traverse all cities'):
      st.write('aaaaaaaaaaaaaaaaaaa')
