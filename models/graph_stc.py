import json
import pandas as pd
import networkx as nx
from geopy.distance import geodesic
import matplotlib.pyplot as plt
import re
from .constants import COLORES_LINEAS_METRO

class MetroGrafoCDMX:
    def __init__(self, ruta_archivo_json):
        self.ruta_archivo = ruta_archivo_json
        self.df_estaciones = None
        self.grafo = None
        
    def cargar_datos_json(self):
        """Carga y procesa el archivo JSON del metro"""
        with open(self.ruta_archivo, 'r', encoding='utf-8') as file:
            metro_json = json.load(file)
        
        estaciones = []
        for feature in metro_json['features']:
            prop = feature['properties']
            coords = feature['geometry']['coordinates']
            
            # Extraer número de estación del CVE_EST
            est_match = re.search(r'(\d+)$', prop.get('CVE_EST', ''))
            est_numero = int(est_match.group(1)) if est_match else 0
            
            estaciones.append({
                'nombre': prop['NOMBRE'],
                'linea': prop['LINEA'], 
                'est': est_numero,  # Número de estación extraído
                'tipo': prop['TIPO'],
                'coordenadas': (coords[1], coords[0]),  # (lat, lon)
                'alcaldia': prop.get('ALCALDIAS', ''),
                'año': prop.get('AÑO', ''),
                'cve_est': prop.get('CVE_EST', ''),
                'sistema': prop.get('SISTEMA', '')
            })
        
        self.df_estaciones = pd.DataFrame(estaciones)
        print(f"Cargadas {len(self.df_estaciones)} estaciones")
        
        # Mostrar resumen por línea
        resumen = self.df_estaciones.groupby('linea').size().sort_index()
        print("Estaciones por línea:")
        for linea, cantidad in resumen.items():
            print(f"  Línea {linea}: {cantidad} estaciones")
        
        return self.df_estaciones
    
    def crear_grafo(self):
        """Crea el grafo conectando solo estaciones consecutivas de cada línea"""[1]
        if self.df_estaciones is None:
            raise ValueError("Primero debes cargar los datos")
        
        self.grafo = nx.Graph()
        
        # Agrupar y ordenar estaciones por línea
        lineas = self.df_estaciones.groupby('linea')
        
        for linea, grupo in lineas:
            # Ordenar por número de estación
            grupo_ordenado = grupo.sort_values(by='est')
            estaciones_linea = grupo_ordenado.to_dict('records')
            
            print(f"Línea {linea}: {len(estaciones_linea)} estaciones ordenadas")
            
            # Agregar nodos
            for estacion in estaciones_linea:
                self.grafo.add_node(
                    estacion['nombre'],
                    **estacion  # Agregar todos los atributos
                )
            
            # Conectar solo estaciones consecutivas
            for i in range(len(estaciones_linea) - 1):
                est_actual = estaciones_linea[i]
                est_siguiente = estaciones_linea[i + 1]
                
                # Calcular distancia real
                coord1 = est_actual['coordenadas']
                coord2 = est_siguiente['coordenadas']
                distancia_km = geodesic(coord1, coord2).kilometers
                
                self.grafo.add_edge(
                    est_actual['nombre'],
                    est_siguiente['nombre'],
                    distancia=distancia_km,
                    tipo='linea',
                    linea=linea
                )
                
                print(f"  Conectado: {est_actual['nombre']} -> {est_siguiente['nombre']} ({distancia_km:.2f} km)")
        
        # Agregar transbordos
        self._agregar_transbordos()
        
        return self.grafo
    
    def _agregar_transbordos(self):
        """Agrega conexiones de transbordo para estaciones con mismo nombre"""[1]
        
        # Agrupar nodos por nombre para encontrar transbordos
        estaciones_por_nombre = {}
        for nodo in self.grafo.nodes():
            # Usar el nombre de la estación para agrupar
            if nodo not in estaciones_por_nombre:
                estaciones_por_nombre[nodo] = []
            estaciones_por_nombre[nodo].append(nodo)
        
        # Buscar estaciones de transbordo en los datos originales
        estaciones_transbordo = self.df_estaciones[
            self.df_estaciones['tipo'].str.contains('Transbordo', na=False)
        ]
        
        transbordos_agregados = 0
        
        # Para cada estación de transbordo, verificar si hay múltiples líneas
        for nombre_estacion in estaciones_transbordo['nombre'].unique():
            # Encontrar todas las entradas de esta estación en diferentes líneas
            lineas_estacion = self.df_estaciones[
                self.df_estaciones['nombre'] == nombre_estacion
            ]
            
            if len(lineas_estacion) > 1:
                # Hay múltiples líneas, crear conexiones de transbordo
                lineas_list = lineas_estacion['linea'].tolist()
                
                # En este caso, como cada entrada es única, marcar como transbordo
                if self.grafo.has_node(nombre_estacion):
                    self.grafo.nodes[nombre_estacion]['es_transbordo'] = True
                    self.grafo.nodes[nombre_estacion]['lineas_transbordo'] = lineas_list
                    transbordos_agregados += 1
                    print(f"  Marcada como transbordo: {nombre_estacion} (Líneas: {', '.join(lineas_list)})")
        
        print(f"Procesadas {transbordos_agregados} estaciones de transbordo")
    
    def verificar_conectividad(self):
        """Verifica la conectividad del grafo"""[1]
        if self.grafo is None:
            return "No hay grafo creado"
        
        # Verificar componentes conectadas
        componentes = list(nx.connected_components(self.grafo))
        
        print(f"El grafo tiene {len(componentes)} componente(s) conectada(s)")
        
        for i, componente in enumerate(componentes):
            print(f"Componente {i+1}: {len(componente)} estaciones")
            if len(componente) < 10:  # Mostrar componentes pequeñas
                print(f"  Estaciones: {list(componente)}")
        
        # Verificar conexiones por línea
        print("\nConexiones por línea:")
        for linea in self.df_estaciones['linea'].unique():
            nodos_linea = [n for n, attr in self.grafo.nodes(data=True) 
                          if attr.get('linea') == linea]
            aristas_linea = [(u, v) for u, v, attr in self.grafo.edges(data=True) 
                            if attr.get('linea') == linea]
            print(f"  Línea {linea}: {len(nodos_linea)} nodos, {len(aristas_linea)} aristas")
        
        return len(componentes) == 1  # True si todo está conectado
    
    def shortest_path(self, origen, destino):
        """Encuentra la ruta más corta entre dos estaciones usando Dijkstra"""[1]
        if self.grafo is None:
            return None, None
        
        try:
            ruta = nx.dijkstra_path(self.grafo, origen, destino, weight='distancia')
            distancia_total = nx.dijkstra_path_length(self.grafo, origen, destino, weight='distancia')
            return ruta, distancia_total
        except nx.NetworkXNoPath:
            print(f"No hay ruta disponible entre {origen} y {destino}")
            return None, None

    def obtener_ruta_mas_corta(self, origen, destino):
        """Encuentra la ruta más corta entre dos estaciones"""[1]
        if self.grafo is None:
            return None, None
        
        try:
            ruta = nx.shortest_path(self.grafo, origen, destino, weight='distancia')
            distancia_total = nx.shortest_path_length(self.grafo, origen, destino, weight='distancia')
            
            # Mostrar detalles de la ruta
            print(f"\nRuta de {origen} a {destino}:")
            print(f"Distancia total: {distancia_total:.2f} km")
            print("Recorrido:")
            
            for i in range(len(ruta)):
                estacion = ruta[i]
                linea = self.grafo.nodes[estacion]['linea']
                es_transbordo = self.grafo.nodes[estacion].get('es_transbordo', False)
                
                transbordo_info = " (TRANSBORDO)" if es_transbordo else ""
                print(f"  {i+1}. {estacion} - Línea {linea}{transbordo_info}")
                
                if i < len(ruta) - 1:
                    siguiente = ruta[i + 1]
                    distancia_tramo = self.grafo[estacion][siguiente]['distancia']
                    print(f"      ↓ {distancia_tramo:.2f} km")
            
            return ruta, distancia_total
            
        except nx.NetworkXNoPath:
            print(f"No hay ruta disponible entre {origen} y {destino}")
            return None, None
        
    def visualizar_grafo(self, figsize=(15, 10), mostrar_etiquetas=True):
        """Visualiza el grafo del metro con colores por línea"""[1]
        if self.grafo is None or self.grafo.number_of_nodes() == 0:
            print("No hay grafo para visualizar")
            return
        
        plt.figure(figsize=figsize)
        
        # Configurar posiciones basadas en coordenadas reales
        pos = {}
        for nodo, attr in self.grafo.nodes(data=True):
            lat, lon = attr['coordenadas']
            pos[nodo] = (lon, lat)  # matplotlib usa (x, y) = (lon, lat)
        

        # Colores oficiales por línea del Metro CDMX
        colores_lineas = COLORES_LINEAS_METRO
        
        # Dibujar nodos por línea
        for linea in set(nx.get_node_attributes(self.grafo, 'linea').values()):
            nodos_linea = [n for n, attr in self.grafo.nodes(data=True) 
                        if attr['linea'] == linea]
            color = colores_lineas.get(linea, '#888888')
            
            # Tamaño diferente para estaciones de transbordo
            nodos_normales = [n for n in nodos_linea 
                            if not self.grafo.nodes[n].get('es_transbordo', False)]
            nodos_transbordo = [n for n in nodos_linea 
                            if self.grafo.nodes[n].get('es_transbordo', False)]
            
            # Dibujar nodos normales
            if nodos_normales:
                nx.draw_networkx_nodes(
                    self.grafo, pos,
                    nodelist=nodos_normales,
                    node_color=color,
                    node_size=150,
                    alpha=0.8,
                    edgecolors='black',
                    linewidths=0.5
                )
            
            # Dibujar nodos de transbordo más grandes
            if nodos_transbordo:
                nx.draw_networkx_nodes(
                    self.grafo, pos,
                    nodelist=nodos_transbordo,
                    node_color=color,
                    node_size=300,
                    alpha=1.0,
                    edgecolors='red',
                    linewidths=2
                )
        
        # Dibujar aristas normales (conexiones de línea)
        aristas_normales = [(u, v) for u, v, attr in self.grafo.edges(data=True) 
                        if attr.get('tipo', 'linea') == 'linea']
        
        # Agrupar aristas por línea para colorearlas
        for linea in set(nx.get_node_attributes(self.grafo, 'linea').values()):
            aristas_linea = [(u, v) for u, v, attr in self.grafo.edges(data=True) 
                            if attr.get('linea') == linea and attr.get('tipo', 'linea') == 'linea']
            color = colores_lineas.get(linea, '#888888')
            
            if aristas_linea:
                nx.draw_networkx_edges(
                    self.grafo, pos,
                    edgelist=aristas_linea,
                    edge_color=color,
                    alpha=0.7,
                    width=2
                )
        
        # Dibujar aristas de transbordo
        aristas_transbordo = [(u, v) for u, v, attr in self.grafo.edges(data=True) 
                            if attr.get('tipo') == 'transbordo']
        if aristas_transbordo:
            nx.draw_networkx_edges(
                self.grafo, pos,
                edgelist=aristas_transbordo,
                edge_color='red',
                width=3,
                alpha=0.8,
                style='dashed'
            )
        
        # Etiquetas para estaciones
        if mostrar_etiquetas:
            # Solo mostrar nombres de estaciones de transbordo para evitar saturación
            estaciones_transbordo = {n: n for n, attr in self.grafo.nodes(data=True) 
                                if attr.get('es_transbordo', False)}
            
            if estaciones_transbordo:
                nx.draw_networkx_labels(
                    self.grafo, pos,
                    labels=estaciones_transbordo,
                    font_size=8,
                    font_weight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8)
                )
        
        plt.title("Grafo del Metro CDMX - Conexiones Consecutivas", fontsize=16, fontweight='bold')
        plt.xlabel("Longitud", fontsize=12)
        plt.ylabel("Latitud", fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Crear leyenda
        legend_elements = []
        for linea, color in colores_lineas.items():
            if any(attr['linea'] == linea for _, attr in self.grafo.nodes(data=True)):
                legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                            markerfacecolor=color, markersize=8, 
                                            label=f'Línea {linea}'))
        
        if legend_elements:
            plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))
        
        plt.tight_layout()
        plt.show()

    def visualizar_linea_especifica(self, linea, figsize=(12, 8)):
        """Visualiza una línea específica del metro"""[1]
        if self.grafo is None:
            print("No hay grafo para visualizar")
            return
        
        # Filtrar nodos de la línea específica
        nodos_linea = [n for n, attr in self.grafo.nodes(data=True) 
                    if attr['linea'] == linea]
        
        if not nodos_linea:
            print(f"No se encontraron estaciones para la línea {linea}")
            return
        
        # Crear subgrafo con solo esa línea
        subgrafo = self.grafo.subgraph(nodos_linea)
        
        plt.figure(figsize=figsize)
        
        # Posiciones basadas en coordenadas
        pos = {}
        for nodo, attr in subgrafo.nodes(data=True):
            lat, lon = attr['coordenadas']
            pos[nodo] = (lon, lat)
        
        # Color de la línea
        colores_lineas = {
            '01': '#FF69B4', '02': '#0000FF', '03': '#00FF00', '04': '#87CEEB',
            '05': '#FFFF00', '06': '#FF0000', '07': '#FFA500', '08': '#008000',
            '09': '#8B4513', '10': '#800080', '11': '#000000', '12': '#FFD700'
        }
        color_linea = colores_lineas.get(linea, '#888888')
        
        # Dibujar nodos
        nx.draw_networkx_nodes(
            subgrafo, pos,
            node_color=color_linea,
            node_size=400,
            alpha=0.8,
            edgecolors='black',
            linewidths=1
        )
        
        # Dibujar aristas
        nx.draw_networkx_edges(
            subgrafo, pos,
            edge_color=color_linea,
            width=3,
            alpha=0.7
        )
        
        # Etiquetas con nombres de estaciones
        nx.draw_networkx_labels(
            subgrafo, pos,
            font_size=10,
            font_weight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9)
        )
        
        plt.title(f"Línea {linea} del Metro CDMX", fontsize=16, fontweight='bold')
        plt.xlabel("Longitud", fontsize=12)
        plt.ylabel("Latitud", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        # Mostrar información de la línea
        print(f"\nInformación de la Línea {linea}:")
        print(f"Total de estaciones: {len(nodos_linea)}")
        
        # Ordenar estaciones por número
        estaciones_ordenadas = sorted(
            [(n, self.grafo.nodes[n]['est']) for n in nodos_linea],
            key=lambda x: x[1]
        )
        
        print("Recorrido de estaciones:")
        distancia_total = 0
        for i, (estacion, num_est) in enumerate(estaciones_ordenadas):
            es_transbordo = self.grafo.nodes[estacion].get('es_transbordo', False)
            transbordo_info = " (TRANSBORDO)" if es_transbordo else ""
            print(f"  {num_est:02d}. {estacion}{transbordo_info}")
            
            if i < len(estaciones_ordenadas) - 1:
                siguiente_estacion = estaciones_ordenadas[i + 1][0]
                if self.grafo.has_edge(estacion, siguiente_estacion):
                    distancia = self.grafo[estacion][siguiente_estacion]['distancia']
                    distancia_total += distancia
                    print(f"       ↓ {distancia:.2f} km")
        
        print(f"\nDistancia total de la línea: {distancia_total:.2f} km")

    def mostrar_estadisticas_visuales(self):
        """Muestra estadísticas del grafo en formato visual"""[1]
        if self.grafo is None:
            print("No hay grafo creado")
            return
        
        # Crear gráficos de estadísticas
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Estaciones por línea
        lineas_count = self.df_estaciones['linea'].value_counts().sort_index()
        ax1.bar(lineas_count.index, lineas_count.values, color='skyblue')
        ax1.set_title('Estaciones por Línea')
        ax1.set_xlabel('Línea')
        ax1.set_ylabel('Número de Estaciones')
        ax1.grid(True, alpha=0.3)
        
        # 2. Distribución de tipos de estación
        tipos_count = self.df_estaciones['tipo'].value_counts()
        ax2.pie(tipos_count.values, labels=tipos_count.index, autopct='%1.1f%%')
        ax2.set_title('Distribución de Tipos de Estación')
        
        # 3. Grado de conectividad
        grados = dict(self.grafo.degree())
        ax3.hist(grados.values(), bins=10, color='lightgreen', alpha=0.7)
        ax3.set_title('Distribución de Conectividad')
        ax3.set_xlabel('Grado (Número de Conexiones)')
        ax3.set_ylabel('Número de Estaciones')
        ax3.grid(True, alpha=0.3)
        
        # 4. Año de construcción
        años_count = self.df_estaciones['año'].value_counts().sort_index()
        ax4.plot(años_count.index, años_count.values, marker='o', linewidth=2, markersize=6)
        ax4.set_title('Estaciones Construidas por Año')
        ax4.set_xlabel('Año')
        ax4.set_ylabel('Número de Estaciones')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

