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