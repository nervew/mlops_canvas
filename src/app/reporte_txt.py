def generar_informe_problemas(data, archivo_salida="informe_problemas.txt"):
    valido = data.get("valido", False)
    detalles = data.get("detalles", {})

    with open(archivo_salida, "w", encoding="utf-8") as f:
        f.write(f"Informe de problemas por columna\n")
        f.write(f"Estado general del dataset: {'Válido' if valido else 'No válido'}\n\n")

        for columna, info in detalles.items():
            aprobado = info.get("aprobado", False)
            problemas = info.get("problemas", [])

            f.write(f"Columna: {columna}\n")
            f.write(f" - Aprobado: {'Sí' if aprobado else 'No'}\n")
            if problemas:
                f.write(f" - Problemas detectados:\n")
                for problema in problemas:
                    f.write(f"    * {problema}\n")
            else:
                f.write(f" - Sin problemas detectados\n")
            f.write("\n")

    print(f"Informe generado en el archivo: {archivo_salida}")

generar_informe_problemas(report)
