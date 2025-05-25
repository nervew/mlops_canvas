import json

def generar_informe(json_path: str, txt_path: str):
    # Cargar JSON
    with open(json_path, 'r', encoding='utf-8') as f:
        reporte = json.load(f)
    
    valido = reporte.get("valido", True)
    detalles = reporte.get("detalles", {})
    
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(f"Estado general del dataset: {'Válido' if valido else 'No válido'}\n\n")
        
        for columna, info in detalles.items():
            aprobado = info.get("aprobado", True)
            problemas = info.get("problemas", [])
            
            f.write(f"Columna: {columna}\n")
            f.write(f"  Aprobado: {'Sí' if aprobado else 'No'}\n")
            if problemas:
                f.write("  Problemas detectados:\n")
                for problema in problemas:
                    f.write(f"    - {problema}\n")
            else:
                f.write("  No se detectaron problemas.\n")
            f.write("\n")

if __name__ == "__main__":
    # Cambia 'reporte.json' por la ruta de tu archivo JSON
    # Cambia 'informe.txt' por la ruta donde quieres guardar el informe
    generar_informe("reporte.json", "informe.txt")
