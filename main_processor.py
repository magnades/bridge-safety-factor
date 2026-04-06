import json
import opensees_model as osm
from types import SimpleNamespace
from vehicle_library import PREDEFINED_VEHICLES

# --- CONFIGURACIÓN DE REFERENCIA ---
# Definimos el vehículo patrón (Portugal/RSA) una sola vez
REFERENCE_VEHICLE = PREDEFINED_VEHICLES["Portugal"]

# --- Configuration Constants ---
INPUT_FILE = "input.json"
OUTPUT_FILE = "output_dummy.json"

def process_BIST_vehicle(json_dict):

    try:
        json_data = SimpleNamespace(**json_dict)
        VehicleType = json_data.VehicleType

        match VehicleType:
            case "1":
                print("Processing Vehicle type 1...")
                F1, F2 = json_data.F1, json_data.F2
                d1, d2, d3 = json_data.D1, json_data.D2, json_data.D3
                return {
                    "name": VehicleType,
                    "n_axles": 2,
                    "axle_loads": [F1, F2],
                    "axle_positions": [0.5*d1+d2+0.5*d3]
                },
            case "2":
                print("Processing Vehicle type 2...")
                F1, F2, F3 = json_data.F1, json_data.F2, json_data.F3
                d1, d2, d3, d4, d5 = json_data.D1, json_data.D2, json_data.D3, json_data.D4, json_data.D5
                return {
                    "name": VehicleType,
                    "n_axles": 3,
                    "axle_loads": [F1, F2, F3],
                    "axle_positions": [0.5*d1+d2+0.5*d3, 0.5*d3+d4+0.5*d5],
                }
            case _:
                raise Exception("Invalid Vehicle Type")

    except Exception as e:
        raise e

def process_BIST_bridge(bridge_data):
    try:
        json_data = SimpleNamespace(**bridge_data)
        bridge_type = json_data.Model
        match bridge_type:
            case "0":
                print("Processing Bridge type 0...")
                properties = {
                    'span_lengths': [json_data.Span],
                    'nodes_per_span': int(json_data.Span/0.2)+1,
                    'support_types': ["second-class", "pinned"],    #["second-class", "pinned", "fixed"]
                    'E': 25000000.0,
                    'A': 0.6,
                    'I': 0.05,
                }
            case "1":
                print("Processing Bridge type 1...")
                properties = {
                    'span_lengths': [json_data.Span],
                    'nodes_per_span': int(json_data.Span/0.2)+1,
                    'support_types': ["second-class", "second-class"],
                    'E': 25000000.0,
                    'A': 0.6,
                    'I': 0.05,
                }
            case "3":
                print("Processing Bridge type 3...")
                span = json_data.Span
                properties = {
                    'span_lengths': [0.8*span, span, 0.8*span],
                    'nodes_per_span': int(json_data.Span/0.2)*3+3,
                    'support_types': ["second-class", "second-class", "second-class", "second-class"],
                    'E': 25000000.0,
                    'A': 0.6,
                    'I': 0.05,
                }
            case _:
                raise Exception("Invalid Bridge Type")
        return properties
    except Exception as e:
        raise e



def calculate_bridge_factors(bridge_data, vehicle_data):
    """
    Función independiente que realiza el análisis de OpenSees para UN puente.
    Retorna los valores máximos de FSM y FSR.
    """
    # 1. Configurar propiedades del puente
    # Asumimos valores estándar de E, A, I si no vienen en el JSON
    properties = process_BIST_bridge(bridge_data)

    # 2. Inicializar vehículos en el diccionario de propiedades
    # Vehículo de Referencia
    osm.create_vehicle(
        properties,
        REFERENCE_VEHICLE["n_axles"],
        REFERENCE_VEHICLE["axle_loads"],
        REFERENCE_VEHICLE["axle_positions"],
        "Portugal"
    )

    # Vehículo del JSON (Especial)
    osm.create_vehicle(
        properties,
  # Asumimos 3 ejes según F1, F2, F3
        vehicle_data["n_axles"],
        vehicle_data["axle_loads"],
        vehicle_data["axle_positions"],
        vehicle_data["name"],
    )

    # 3. Ejecutar análisis de carga móvil (Envolventes)
    osm.run_analysis(properties)

    # 4. Calcular Ratios (FSM y FSR)
    # Según tu lógica anterior, esto compara 'Analisis' vs 'Portugal'
    osm.compute_ratios_between_vehicles(
        properties,
        ref_vehicle="Portugal",
        new_vehicle=vehicle_data["name"]
    )

    # 5. Extraer Máximos Absolutos
    global_fs = osm.compute_global_fs(properties, vehicle_data["name"])

    return global_fs['FSM'], global_fs['FSV']


def process_api_request(json_data):
    """
    Simula la recepción de la API y procesa la lista de puentes.
    """
    vehicle_info = process_BIST_vehicle(api_input)

    print(f"--- Iniciando procesamiento para Vehículo: {vehicle_info['name']} ---")

    for bridge in json_data["Bridges"]:
        tag = bridge["Tag"]
        print(f"Analizando Puente: {tag} (Vano: {bridge['Span']}m)...")

        # Llamada a la función de cálculo
        fsm, fsr = calculate_bridge_factors(bridge, vehicle_info)

        # Actualizar el objeto JSON con los resultados
        bridge["FSM"] = round(fsm, 3)
        bridge["FSR"] = round(fsr, 3)

    return json_data


# --- SIMULACIÓN DE EJECUCIÓN ---

with open(INPUT_FILE, 'r') as f:
    api_input = json.load(f)

# Procesar
final_json = process_api_request(api_input)

# Mostrar resultado final listo para enviar de vuelta
print("\n--- Procesamiento Finalizado ---")
print(json.dumps(final_json, indent=2))