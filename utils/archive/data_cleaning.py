import os
import json

extracted_data_path = "../data/extracted_data"
working_data_path = "../data/working_data"


def merge_json_files(folder_path, output_folder):
    final_data = []
    for filename in os.listdir(folder_path):  # ← Corregido
        if filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                extracted_data = json.load(file)
                if "elementList" in extracted_data:
                    final_data += extracted_data["elementList"]
    os.makedirs(output_folder, exist_ok=True)  # ← Asegura que la carpeta exista
    with open(os.path.join(output_folder, "merged_data.json"), 'w', encoding='utf-8') as outfile:
        json.dump(final_data, outfile, indent=4, ensure_ascii=False)
    return final_data
    
merge_json_files(extracted_data_path, working_data_path)

def merge_json_files_2(folder_path, output_folder):
    final_data = []

    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                extracted_data = json.load(file)

                # Caso 1: Idealista API -> dict con "elementList"
                if isinstance(extracted_data, dict) and "elementList" in extracted_data:
                    final_data.extend(extracted_data["elementList"])

                # Caso 2: archivo ya es una lista
                elif isinstance(extracted_data, list):
                    final_data.extend(extracted_data)

                # Caso raro: dict sin "elementList"
                else:
                    print(f"⚠️ Ojo: {filename} no tiene 'elementList' ni es lista. Ignorado.")

    # Asegura que la carpeta exista
    os.makedirs(output_folder, exist_ok=True)

    # Guardar mergeado
    output_file = os.path.join(output_folder, "merged_data_20250819.json")
    with open(output_file, 'w', encoding='utf-8') as outfile:
        json.dump(final_data, outfile, indent=4, ensure_ascii=False)

    print(f"✅ Archivos combinados en {output_file}. Total propiedades: {len(final_data)}")
    return final_data