import json

def generar_informe(json_path: str, txt_path: str):
    with open(json_path, 'r', encoding='utf-8') as f:
        reporte = json.load(f)
    
    valido = reporte.get("valido", True)
    detalles = reporte.get("detalles", {})
    
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(f"Estado general del dataset: {'Válido' if valido else 'No válido'}\n\n")
        
        for columna, info in detalles.items():
            aprobado = info.get("aprobado", True)
            problemas = info.get("problemas", [])
            
            # Solo columnas con problemas (no aprobadas)
            if not aprobado and problemas:
                f.write(f"Columna: {columna}\n")
                f.write(f"  Aprobado: No\n")
                f.write("  Problemas detectados:\n")
                for problema in problemas:
                    f.write(f"    - {problema}\n")
                f.write("\n")

if __name__ == "__main__":
    generar_informe("reporte.json", "informe.txt")
