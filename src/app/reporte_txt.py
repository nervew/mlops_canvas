import json

def generar_informe(json_path: str, txt_path: str):
    with open(json_path, 'r', encoding='utf-8') as f:
        reporte = json.load(f)
    
    valido = reporte.get("valido", True)
    detalles = reporte.get("detalles", {})

    # Filtrar columnas que tienen problemas
    columnas_con_problemas = {col: info for col, info in detalles.items() if info.get("problemas")}
    
    # Ordenar columnas alfabéticamente
    columnas_ordenadas = sorted(columnas_con_problemas.keys())
    
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(f"Estado general del dataset: {'Válido' if valido else 'No válido'}\n\n")
        
        for columna in columnas_ordenadas:
            info = columnas_con_problemas[columna]
            aprobado = info.get("aprobado", True)
            problemas = info.get("problemas", [])
            
            f.write(f"Columna: {columna}\n")
            f.write(f"  Aprobado: {'Sí' if aprobado else 'No'}\n")
            f.write("  Problemas detectados:\n")
            for problema in problemas:
                f.write(f"    - {problema}\n")
            f.write("\n")

if __name__ == "__main__":
    generar_informe("reporte.json", "informe.txt")
