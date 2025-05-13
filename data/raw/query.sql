SELECT
  -- Agrupa por el primer día del mes (TIMESTAMP). Ej: 2025-04-01 00:00:00
  date_trunc('month', FechaRegistro) AS mes_registro,
 
  -- Spark SQL optimiza COUNT(DISTINCT STRUCT(...)) más que CONCAT + DISTINCT
  COUNT(DISTINCT struct(TipoIDPaciente, NoIDPaciente)) AS usuarios_unicos
 
FROM  hive_metastore.db_mipres_suministro.dbo_tsum_tx
 
-- (Opcional) Filtra un rango de fechas para reducir datos escaneados
-- WHERE fecha_registro BETWEEN '2024-01-01' AND current_date()
 
GROUP BY
  mes_registro
 
ORDER BY
  mes_registro;