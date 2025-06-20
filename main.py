from models.graph_stc import MetroGrafoCDMX
from models.tsp_stc import MetroGrafoCDMXWithTSP
import json

# Función de uso principal
def procesar_metro_cdmx(ruta_archivo_json):
    """Función principal corregida para procesar el metro"""
    metro = MetroGrafoCDMX(ruta_archivo_json)
    # Cargar datos
    print("=== Cargando datos ===")
    metro.cargar_datos_json()
    
    # Crear grafo corregido
    print("\n=== Creando grafo con conexiones consecutivas ===")
    metro.crear_grafo()
    
    # Verificar conectividad
    print("\n=== Verificando conectividad ===")
    metro.verificar_conectividad()
    
    # Estadísticas
    print(f"\n=== Estadísticas ===")
    print(f"Nodos: {metro.grafo.number_of_nodes()}")
    print(f"Aristas: {metro.grafo.number_of_edges()}")

    # Visualizar el grafo completo
    #print("=== Visualizando grafo completo ===")
    #metro.visualizar_grafo()

    return metro


def obtener_mejor_estacion_inicio(metro, tsp_solver):
    estaciones = list(metro.grafo.nodes())
    tsp_solver.agregar_tiempos_al_grafo()
    mejor_estacion = None
    mejor_tiempo = float('inf')
    resultados = []
    archivo_resultados = "resultados_estaciones.json"
    for i, estacion_inicio in enumerate(estaciones):
        print(f"Procesando estación de inicio: {estacion_inicio} ({i+1}/{len(estaciones)})")
        ruta, peso, stats = tsp_solver.resolver_tsp_optimizado(estacion_inicio, "combinado")
        ruta_regreso, distancia_regreso = metro.shortest_path(ruta[-2], ruta[-1])
        tiempo_regreso = distancia_regreso / tsp_solver.velocidad_promedio_kmh

        tiempo_total = stats['tiempo_total_horas'] + tiempo_regreso
        distancia_total = stats['distancia_total_km'] + distancia_regreso
        ruta_completa = ruta[:-2] + ruta_regreso

        resultado = {
            "estacion_inicio": estacion_inicio,
            "ruta_completa": ruta_completa,
            "tiempo_total_horas": tiempo_total,
            "distancia_total_km": distancia_total
        }
        resultados.append(resultado)

        if tiempo_total < mejor_tiempo:
            mejor_tiempo = tiempo_total
            mejor_estacion = estacion_inicio

        # Guardar resultados en JSON en cada iteración
        with open(archivo_resultados, "w", encoding='utf-8') as f:
            json.dump(resultados, f, indent=4, ensure_ascii=False)

    return mejor_estacion, resultados

def resolver_tsp_optimizado(metro, tsp_solver, estacion_inicio):
    """
    Resuelve el TSP optimizado para una estación de inicio dada y un tipo de ruta.
    """
    ruta, peso, stats = tsp_solver.resolver_tsp_optimizado(estacion_inicio, "combinado")
    ruta_regreso, distancia_regreso = metro.shortest_path(ruta[-2], ruta[-1])
    tiempo_regreso = distancia_regreso / tsp_solver.velocidad_promedio_kmh
    tiempo_total = stats['tiempo_total_horas'] + tiempo_regreso
    distancia_total = stats['distancia_total_km'] + distancia_regreso
    ruta_completa = ruta[:-2] + ruta_regreso
    return ruta_completa, tiempo_total, distancia_total



# Ejemplo de uso
if __name__ == "__main__":
    metro = procesar_metro_cdmx("STC_Metro_estaciones_utm14n.json")
    tsp_solver = MetroGrafoCDMXWithTSP(metro)
    tsp_solver.agregar_tiempos_al_grafo()

    #mejor_estacion, resultados = obtener_mejor_estacion_inicio(metro, tsp_solver)
    #print(f"\nMejor estación de inicio: {mejor_estacion}")
    estacion_inicio = "Centro Médico"  # Cambia esto a la estación de inicio deseada
    ruta_completa, tiempo_total, distancia_total = resolver_tsp_optimizado(metro, tsp_solver, estacion_inicio)
    #print (f"Ruta completa: {ruta_completa}")
    print(f"Tiempo total: {tiempo_total:.2f} horas")
    print(f"Distancia total: {distancia_total:.2f} km")
    tsp_solver.visualizar_ruta_tsp(ruta_completa, distancia_total)



