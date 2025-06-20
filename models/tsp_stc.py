import networkx as nx
from functools import lru_cache
import matplotlib.pyplot as plt
from .constants import COLORES_LINEAS_METRO
import networkx as nx
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import time


class MetroGrafoCDMXWithTSP:
    def __init__(self, metro_grafo_cdmx):
        self.grafo = metro_grafo_cdmx.grafo
        self.df_estaciones = metro_grafo_cdmx.df_estaciones
        self.metro_original = metro_grafo_cdmx
        # Velocidad promedio real del Metro CDMX según datos oficiales
        self.velocidad_promedio_kmh = 21.6  # km/h actual según reportes 2022[2][5]
        self.velocidad_objetivo_kmh = 36.0   # km/h velocidad comercial objetivo[3]

    def resolver_tsp(self, estacion_inicio_nombre, heuristico=True):
        """Resuelve el TSP usando programación dinámica con memoización, estación de inicio por nombre"""
        if self.grafo is None:
            raise ValueError("El grafo no está creado")

        nodos = list(self.grafo.nodes())
        if estacion_inicio_nombre not in nodos:
            raise ValueError(f"La estación {estacion_inicio_nombre} no está en el grafo")

        n = len(nodos)
        print(f"Resolviendo TSP para {n} estaciones, iniciando en: {estacion_inicio_nombre}")
        
        if n > 20:
            print(f"ADVERTENCIA: TSP con {n} nodos puede ser muy lento. Considera usar una heurística.")

        if heuristico:
            return self.resolver_tsp_heuristico(estacion_inicio_nombre)

        indice_inicio = nodos.index(estacion_inicio_nombre)

        # Crear matriz de distancias usando rutas más cortas
        print("Calculando matriz de distancias...")
        matriz_distancias = self._crear_matriz_distancias(nodos)

        @lru_cache(None)
        def tsp(mask, pos):
            if mask == (1 << n) - 1:
                return matriz_distancias[pos][indice_inicio]

            ans = float('inf')
            for city in range(n):
                if (mask & (1 << city)) == 0:
                    ans = min(ans, matriz_distancias[pos][city] + tsp(mask | (1 << city), city))
            return ans

        # Reconstruir ruta
        def reconstruir_ruta():
            mask = 1 << indice_inicio
            pos = indice_inicio
            ruta = [nodos[indice_inicio]]

            while mask != (1 << n) - 1:
                siguiente = None
                mejor_costo = float('inf')
                for city in range(n):
                    if (mask & (1 << city)) == 0:
                        costo = matriz_distancias[pos][city] + tsp(mask | (1 << city), city)
                        if costo < mejor_costo:
                            mejor_costo = costo
                            siguiente = city
                ruta.append(nodos[siguiente])
                pos = siguiente
                mask |= (1 << siguiente)

            ruta.append(nodos[indice_inicio])  # Regresar al inicio
            return ruta

        print("Ejecutando algoritmo TSP...")
        distancia_minima = tsp(1 << indice_inicio, indice_inicio)
        ruta_optima = reconstruir_ruta()

        return ruta_optima, distancia_minima

    def _crear_matriz_distancias(self, nodos):
        """Crea matriz de distancias usando rutas más cortas entre todos los nodos"""
        n = len(nodos)
        matriz_distancias = [[float('inf')] * n for _ in range(n)]
        
        # Calcular rutas más cortas entre todos los pares de nodos
        rutas_cortas = dict(nx.all_pairs_dijkstra_path_length(self.grafo, weight='distancia'))
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    matriz_distancias[i][j] = 0
                else:
                    try:
                        matriz_distancias[i][j] = rutas_cortas[nodos[i]][nodos[j]]
                    except KeyError:
                        matriz_distancias[i][j] = float('inf')
        
        return matriz_distancias

    def resolver_tsp_heuristico(self, estacion_inicio_nombre):
        """Resuelve TSP usando heurística del vecino más cercano para grafos grandes"""
        if estacion_inicio_nombre not in self.grafo.nodes():
            raise ValueError(f"La estación {estacion_inicio_nombre} no está en el grafo")

        nodos = list(self.grafo.nodes())
        visitados = set()
        ruta = [estacion_inicio_nombre]
        visitados.add(estacion_inicio_nombre)
        distancia_total = 0
        actual = estacion_inicio_nombre

        print(f"Usando heurística del vecino más cercano para {len(nodos)} estaciones...")

        while len(visitados) < len(nodos):
            mejor_siguiente = None
            mejor_distancia = float('inf')

            for nodo in nodos:
                if nodo not in visitados:
                    try:
                        distancia = nx.shortest_path_length(
                            self.grafo, actual, nodo, weight='distancia'
                        )
                        if distancia < mejor_distancia:
                            mejor_distancia = distancia
                            mejor_siguiente = nodo
                    except nx.NetworkXNoPath:
                        continue

            if mejor_siguiente is None:
                print("No se puede completar el tour - grafo no conectado")
                break

            ruta.append(mejor_siguiente)
            visitados.add(mejor_siguiente)
            distancia_total += mejor_distancia
            actual = mejor_siguiente

        # Regresar al inicio
        try:
            distancia_regreso = nx.shortest_path_length(
                self.grafo, actual, estacion_inicio_nombre, weight='distancia'
            )
            ruta.append(estacion_inicio_nombre)
            distancia_total += distancia_regreso
        except nx.NetworkXNoPath:
            print("No se puede regresar al punto de inicio")

        return ruta, distancia_total

    def visualizar_ruta_tsp(self, ruta, distancia_total, figsize=(15, 10)):
        """Visualiza la ruta TSP en el mapa del metro"""
        if not ruta or len(ruta) < 2:
            print("No hay ruta válida para visualizar")
            return

        plt.figure(figsize=figsize)

        # Posiciones basadas en coordenadas
        pos = {}
        for nodo, attr in self.grafo.nodes(data=True):
            lat, lon = attr['coordenadas']
            pos[nodo] = (lon, lat)

        # Colores por línea
        colores_lineas = COLORES_LINEAS_METRO

        # Dibujar todas las estaciones con transparencia
        for linea in set(nx.get_node_attributes(self.grafo, 'linea').values()):
            nodos_linea = [n for n, attr in self.grafo.nodes(data=True) 
                          if attr['linea'] == linea]
            color = colores_lineas.get(linea, '#888888')

            nx.draw_networkx_nodes(
                self.grafo, pos,
                nodelist=nodos_linea,
                node_color=color,
                node_size=100,
                alpha=0.3
            )

        # Dibujar todas las conexiones con transparencia
        nx.draw_networkx_edges(
            self.grafo, pos,
            alpha=0.2,
            width=1,
            edge_color='gray'
        )

        # Resaltar la ruta TSP
        for i in range(len(ruta) - 1):
            estacion_actual = ruta[i]
            estacion_siguiente = ruta[i + 1]

            # Dibujar nodos de la ruta
            nx.draw_networkx_nodes(
                self.grafo, pos,
                nodelist=[estacion_actual],
                node_color='red',
                node_size=300,
                alpha=0.8
            )

            # Dibujar la ruta más corta entre estaciones consecutivas en TSP
            try:
                camino = nx.shortest_path(
                    self.grafo, estacion_actual, estacion_siguiente, weight='distancia'
                )
                
                # Dibujar el camino
                for j in range(len(camino) - 1):
                    nx.draw_networkx_edges(
                        self.grafo, pos,
                        edgelist=[(camino[j], camino[j + 1])],
                        edge_color='red',
                        width=3,
                        alpha=0.8
                    )
            except nx.NetworkXNoPath:
                # Si no hay camino directo, dibujar línea punteada
                plt.plot([pos[estacion_actual][0], pos[estacion_siguiente][0]],
                        [pos[estacion_actual][1], pos[estacion_siguiente][1]],
                        'r--', linewidth=2, alpha=0.5)

        # Marcar estación de inicio/fin
        estacion_inicio = ruta[0]
        nx.draw_networkx_nodes(
            self.grafo, pos,
            nodelist=[estacion_inicio],
            node_color='green',
            node_size=500,
            alpha=1.0
        )

        # Etiquetas para estaciones en la ruta
        etiquetas_ruta = {estacion: f"{i+1}" for i, estacion in enumerate(ruta[:-1])}
        nx.draw_networkx_labels(
            self.grafo, pos,
            labels=etiquetas_ruta,
            font_size=8,
            font_weight='bold'
        )

        plt.title(f"Ruta TSP del Metro CDMX\nDistancia Total: {distancia_total:.2f} km", 
                 fontsize=16, fontweight='bold')
        plt.xlabel("Longitud")
        plt.ylabel("Latitud")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def mostrar_detalles_ruta(self, ruta, distancia_total):
        """Muestra detalles de la ruta TSP"""
        if not ruta:
            print("No hay ruta para mostrar")
            return

        print(f"\n{'='*60}")
        print(f"RUTA TSP ÓPTIMA DEL METRO CDMX")
        print(f"{'='*60}")
        print(f"Estación de inicio/fin: {ruta[0]}")
        print(f"Total de estaciones visitadas: {len(ruta)-1}")
        print(f"Distancia total: {distancia_total:.2f} km")
        print(f"{'='*60}")

        distancia_acumulada = 0
        for i in range(len(ruta) - 1):
            actual = ruta[i]
            siguiente = ruta[i + 1]

            # Obtener información de la estación
            info_actual = self.grafo.nodes[actual]
            linea = info_actual['linea']
            es_transbordo = info_actual.get('es_transbordo', False)

            try:
                distancia_tramo = nx.shortest_path_length(
                    self.grafo, actual, siguiente, weight='distancia'
                )
                distancia_acumulada += distancia_tramo

                transbordo_info = " 🔄" if es_transbordo else ""
                print(f"{i+1:2d}. {actual} (Línea {linea}){transbordo_info}")
                if i < len(ruta) - 2:  # No mostrar flecha para el último
                    print(f"    ↓ {distancia_tramo:.2f} km (Acumulado: {distancia_acumulada:.2f} km)")

            except nx.NetworkXNoPath:
                print(f"{i+1:2d}. {actual} (Línea {linea}) - SIN CONEXIÓN DIRECTA")

        print(f"{'='*60}")

    def resolver_tsp_por_linea(self, linea):
        """Resuelve TSP solo para las estaciones de una línea específica"""
        nodos_linea = [n for n, attr in self.grafo.nodes(data=True) 
                      if attr['linea'] == linea]
        
        if len(nodos_linea) < 2:
            print(f"La línea {linea} no tiene suficientes estaciones")
            return None, None

        # Crear subgrafo de la línea
        subgrafo = self.grafo.subgraph(nodos_linea)
        
        # Tomar la primera estación como inicio
        estacion_inicio = nodos_linea[0]
        
        print(f"Resolviendo TSP para Línea {linea} ({len(nodos_linea)} estaciones)")
        
        # Usar el algoritmo TSP en el subgrafo
        metro_temp = type('obj', (object,), {'grafo': subgrafo, 'df_estaciones': self.df_estaciones})
        tsp_temp = MetroGrafoCDMXWithTSP(metro_temp)
        
        return tsp_temp.resolver_tsp(estacion_inicio)

    def agregar_tiempos_al_grafo(self, usar_velocidad_objetivo=False):
        """Agrega tiempos a las aristas del grafo basándose en distancia y velocidad"""
        velocidad = self.velocidad_objetivo_kmh if usar_velocidad_objetivo else self.velocidad_promedio_kmh
        
        print(f"Agregando tiempos con velocidad: {velocidad} km/h")
        
        for u, v, data in self.grafo.edges(data=True):
            distancia = data.get('distancia', None)
            if distancia is not None:
                # Tiempo en horas
                tiempo_horas = distancia / velocidad
                # Tiempo en minutos
                tiempo_minutos = tiempo_horas * 60
                # Agregar tiempo de parada en estaciones (promedio 30 segundos)
                tiempo_parada = 0.5  # minutos
                
                data['tiempo_minutos'] = tiempo_minutos + tiempo_parada
                data['tiempo_horas'] = tiempo_horas + (tiempo_parada / 60)
                
                # Crear peso combinado (distancia + tiempo normalizado)
                # Normalizar tiempo a escala de distancia (1 hora = 1 km equivalente)
                data['peso_combinado'] = distancia + tiempo_horas
        
        print(f"Tiempos agregados a {self.grafo.number_of_edges()} aristas")
        return self.grafo

    def encontrar_mejor_estacion_inicio(self, criterio='combinado', top_n=5):
        """Encuentra la(s) mejor(es) estación(es) de inicio que minimice distancia y tiempo"""
        estaciones = list(self.grafo.nodes())
        resultados = []
        
        print(f"Evaluando {len(estaciones)} estaciones como punto de inicio...")
        print(f"Criterio de optimización: {criterio}")
        
        for i, estacion in enumerate(estaciones):
            if i % 20 == 0:
                print(f"Progreso: {i}/{len(estaciones)} estaciones evaluadas")
            
            suma_distancias = 0
            suma_tiempos = 0
            suma_combinado = 0
            conexiones_validas = 0
            
            for destino in estaciones:
                if estacion != destino:
                    try:
                        # Calcular distancia
                        distancia = nx.shortest_path_length(
                            self.grafo, estacion, destino, weight='distancia'
                        )
                        
                        # Calcular tiempo
                        if 'tiempo_minutos' in list(self.grafo.edges(data=True))[0][2]:
                            tiempo_minutos = nx.shortest_path_length(
                                self.grafo, estacion, destino, weight='tiempo_minutos'
                            )
                        else:
                            tiempo_minutos = (distancia / self.velocidad_promedio_kmh) * 60
                        
                        # Calcular peso combinado
                        peso_comb = nx.shortest_path_length(
                            self.grafo, estacion, destino, weight='peso_combinado'
                        ) if 'peso_combinado' in list(self.grafo.edges(data=True))[0][2] else distancia + (tiempo_minutos / 60)
                        
                        suma_distancias += distancia
                        suma_tiempos += tiempo_minutos
                        suma_combinado += peso_comb
                        conexiones_validas += 1
                        
                    except nx.NetworkXNoPath:
                        continue
            
            if conexiones_validas > 0:
                # Obtener información adicional de la estación
                info_estacion = self.grafo.nodes[estacion]
                es_transbordo = info_estacion.get('es_transbordo', False)
                linea = info_estacion.get('linea', 'N/A')
                
                resultado = {
                    'estacion': estacion,
                    'linea': linea,
                    'es_transbordo': es_transbordo,
                    'suma_distancias': suma_distancias,
                    'suma_tiempos_minutos': suma_tiempos,
                    'suma_tiempos_horas': suma_tiempos / 60,
                    'suma_combinado': suma_combinado,
                    'conexiones_validas': conexiones_validas,
                    'promedio_distancia': suma_distancias / conexiones_validas,
                    'promedio_tiempo_minutos': suma_tiempos / conexiones_validas,
                    'promedio_combinado': suma_combinado / conexiones_validas
                }
                resultados.append(resultado)
        
        # Ordenar según criterio
        if criterio == 'distancia':
            resultados.sort(key=lambda x: x['suma_distancias'])
        elif criterio == 'tiempo':
            resultados.sort(key=lambda x: x['suma_tiempos_minutos'])
        else:  # criterio == 'combinado'
            resultados.sort(key=lambda x: x['suma_combinado'])
        
        # Mostrar resultados
        print(f"\n{'='*80}")
        print(f"TOP {top_n} MEJORES ESTACIONES DE INICIO (Criterio: {criterio.upper()})")
        print(f"{'='*80}")
        
        for i, resultado in enumerate(resultados[:top_n]):
            transbordo_info = " 🔄 TRANSBORDO" if resultado['es_transbordo'] else ""
            print(f"\n{i+1}. {resultado['estacion']} (Línea {resultado['linea']}){transbordo_info}")
            print(f"   Distancia total: {resultado['suma_distancias']:.2f} km")
            print(f"   Tiempo total: {resultado['suma_tiempos_horas']:.2f} horas ({resultado['suma_tiempos_minutos']:.1f} min)")
            print(f"   Peso combinado: {resultado['suma_combinado']:.2f}")
            print(f"   Promedios: {resultado['promedio_distancia']:.2f} km, {resultado['promedio_tiempo_minutos']:.1f} min")
        
        return resultados[:top_n]

    def resolver_tsp_optimizado(self, estacion_inicio_nombre, criterio_peso='combinado'):
        """Resuelve TSP usando diferentes criterios de peso (distancia, tiempo, combinado)"""
        if estacion_inicio_nombre not in self.grafo.nodes():
            raise ValueError(f"La estación {estacion_inicio_nombre} no está en el grafo")
        
        # Asegurar que los tiempos están agregados
        if 'tiempo_minutos' not in list(self.grafo.edges(data=True))[0][2]:
            self.agregar_tiempos_al_grafo()
        
        # Seleccionar peso según criterio
        peso_mapa = {
            'distancia': 'distancia',
            'tiempo': 'tiempo_minutos', 
            'combinado': 'peso_combinado'
        }
        
        peso_attr = peso_mapa.get(criterio_peso, 'peso_combinado')
        
        nodos = list(self.grafo.nodes())
        n = len(nodos)
        indice_inicio = nodos.index(estacion_inicio_nombre)
        
        print(f"Resolviendo TSP optimizado para {n} estaciones")
        print(f"Estación inicio: {estacion_inicio_nombre}")
        print(f"Criterio de optimización: {criterio_peso}")
        

        # Crear matriz de distancias/tiempos
        matriz_pesos = self._crear_matriz_pesos(nodos, peso_attr)

        if n > 15:
            print("Usando heurística para grafo grande...")
            return self._resolver_tsp_heuristico_optimizado(estacion_inicio_nombre, peso_attr)

        
        # Resolver TSP con programación dinámica
        from functools import lru_cache
        
        @lru_cache(None)
        def tsp(mask, pos):
            if mask == (1 << n) - 1:
                return matriz_pesos[pos][indice_inicio]
            
            ans = float('inf')
            for city in range(n):
                if (mask & (1 << city)) == 0:
                    ans = min(ans, matriz_pesos[pos][city] + tsp(mask | (1 << city), city))
            return ans
        
        def reconstruir_ruta():
            mask = 1 << indice_inicio
            pos = indice_inicio
            ruta = [nodos[indice_inicio]]
            
            while mask != (1 << n) - 1:
                siguiente = None
                mejor_costo = float('inf')
                for city in range(n):
                    if (mask & (1 << city)) == 0:
                        costo = matriz_pesos[pos][city] + tsp(mask | (1 << city), city)
                        if costo < mejor_costo:
                            mejor_costo = costo
                            siguiente = city
                
                ruta.append(nodos[siguiente])
                pos = siguiente
                mask |= (1 << siguiente)
            
            ruta.append(nodos[indice_inicio])
            return ruta
        
        print("Ejecutando algoritmo TSP optimizado...")
        inicio_tiempo = time.time()
        
        peso_minimo = tsp(1 << indice_inicio, indice_inicio)
        ruta_optima = reconstruir_ruta()
        
        tiempo_ejecucion = time.time() - inicio_tiempo
        print(f"TSP resuelto en {tiempo_ejecucion:.2f} segundos")
        
        # Calcular estadísticas completas
        estadisticas = self._calcular_estadisticas_ruta(ruta_optima)
        
        return ruta_optima, peso_minimo, estadisticas

    def _crear_matriz_pesos(self, nodos, peso_attr):
        """Crea matriz de pesos para TSP"""
        n = len(nodos)
        matriz = [[float('inf')] * n for _ in range(n)]
        
        # Calcular rutas más cortas con el peso especificado
        rutas_cortas = dict(nx.all_pairs_dijkstra_path_length(self.grafo, weight=peso_attr))
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    matriz[i][j] = 0
                else:
                    try:
                        matriz[i][j] = rutas_cortas[nodos[i]][nodos[j]]
                    except KeyError:
                        matriz[i][j] = float('inf')
        print("Matriz de pesos creada:")
        print(matriz)
        return matriz

    def _resolver_tsp_heuristico_optimizado(self, estacion_inicio, peso_attr):
        """Versión optimizada de la heurística para grafos grandes"""
        nodos = list(self.grafo.nodes())
        visitados = set([estacion_inicio])
        ruta = [estacion_inicio]
        peso_total = 0
        actual = estacion_inicio
        
        while len(visitados) < len(nodos):
            mejor_siguiente = None
            mejor_peso = float('inf')
            
            for nodo in nodos:
                if nodo not in visitados:
                    try:
                        peso = nx.shortest_path_length(
                            self.grafo, actual, nodo, weight=peso_attr
                        )
                        if peso < mejor_peso:
                            mejor_peso = peso
                            mejor_siguiente = nodo
                    except nx.NetworkXNoPath:
                        continue
            
            if mejor_siguiente is None:
                break
                
            ruta.append(mejor_siguiente)
            visitados.add(mejor_siguiente)
            peso_total += mejor_peso
            actual = mejor_siguiente
        
        # Regresar al inicio
        try:
            peso_regreso = nx.shortest_path_length(
                self.grafo, actual, estacion_inicio, weight=peso_attr
            )
            ruta.append(estacion_inicio)
            peso_total += peso_regreso
        except nx.NetworkXNoPath:
            print("No se puede regresar al inicio")
        
        estadisticas = self._calcular_estadisticas_ruta(ruta)
        return ruta, peso_total, estadisticas

    def _calcular_estadisticas_ruta(self, ruta):
        """Calcula estadísticas completas de una ruta"""
        if len(ruta) < 2:
            return {}
        
        distancia_total = 0
        tiempo_total_minutos = 0
        tiempo_total_horas = 0
        
        for i in range(len(ruta) - 1):
            actual = ruta[i]
            siguiente = ruta[i + 1]
            
            try:
                # Distancia
                dist = nx.shortest_path_length(
                    self.grafo, actual, siguiente, weight='distancia'
                )
                distancia_total += dist
                
                # Tiempo
                if 'tiempo_minutos' in list(self.grafo.edges(data=True))[0][2]:
                    tiempo_min = nx.shortest_path_length(
                        self.grafo, actual, siguiente, weight='tiempo_minutos'
                    )
                    tiempo_total_minutos += tiempo_min
                else:
                    tiempo_total_minutos += (dist / self.velocidad_promedio_kmh) * 60
                    
            except nx.NetworkXNoPath:
                continue
        
        tiempo_total_horas = tiempo_total_minutos / 60
        
        return {
            'distancia_total_km': distancia_total,
            'tiempo_total_minutos': tiempo_total_minutos,
            'tiempo_total_horas': tiempo_total_horas,
            'estaciones_visitadas': len(ruta) - 1,
            'velocidad_promedio_kmh': distancia_total / tiempo_total_horas if tiempo_total_horas > 0 else 0
        }

    def comparar_velocidades(self, ruta):
        """Compara tiempos de ruta con velocidad actual vs velocidad objetivo"""
        if len(ruta) < 2:
            return
        
        distancia_total = 0
        for i in range(len(ruta) - 1):
            try:
                dist = nx.shortest_path_length(
                    self.grafo, ruta[i], ruta[i+1], weight='distancia'
                )
                distancia_total += dist
            except nx.NetworkXNoPath:
                continue
        
        tiempo_actual = (distancia_total / self.velocidad_promedio_kmh) * 60  # minutos
        tiempo_objetivo = (distancia_total / self.velocidad_objetivo_kmh) * 60  # minutos
        
        print(f"\n{'='*60}")
        print(f"COMPARACIÓN DE VELOCIDADES")
        print(f"{'='*60}")
        print(f"Distancia total: {distancia_total:.2f} km")
        print(f"Velocidad actual ({self.velocidad_promedio_kmh} km/h): {tiempo_actual:.1f} minutos ({tiempo_actual/60:.2f} horas)")
        print(f"Velocidad objetivo ({self.velocidad_objetivo_kmh} km/h): {tiempo_objetivo:.1f} minutos ({tiempo_objetivo/60:.2f} horas)")
        print(f"Diferencia: {tiempo_actual - tiempo_objetivo:.1f} minutos")
        print(f"Tiempo adicional por velocidad reducida: {((tiempo_actual/tiempo_objetivo - 1) * 100):.1f}%")

    def optimizar_estacion_inicio_automatico(self):
        """Encuentra automáticamente la mejor estación de inicio para TSP"""
        print("Buscando la mejor estación de inicio...")
        
        # Agregar tiempos si no existen
        if 'tiempo_minutos' not in list(self.grafo.edges(data=True))[0][2]:
            self.agregar_tiempos_al_grafo()
        
        # Encontrar mejores estaciones
        mejores_estaciones = self.encontrar_mejor_estacion_inicio(criterio='combinado', top_n=3)
        
        if not mejores_estaciones:
            print("No se encontraron estaciones válidas")
            return None
        
        mejor_estacion = mejores_estaciones[0]['estacion']
        print(f"\nMejor estación de inicio encontrada: {mejor_estacion}")
        
        # Resolver TSP desde la mejor estación
        ruta, peso, stats = self.resolver_tsp_optimizado(mejor_estacion, 'combinado')
        
        return {
            'estacion_optima': mejor_estacion,
            'ruta_optima': ruta,
            'peso_total': peso,
            'estadisticas': stats,
            'alternativas': mejores_estaciones[1:3]
        }