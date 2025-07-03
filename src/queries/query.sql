SELECT
  -- 1. Agrupar por mes
  date_trunc('month', FechaRegistro) AS mes_registro,

  -- 2. Total de usuarios únicos
  COUNT(DISTINCT struct(TipoIDPaciente, NoIDPaciente)) AS usuarios_unicos,

  -- 3. Usuarios cuyo código de EPS comienza por 'CCF'
  COUNT(DISTINCT CASE 
      WHEN CodigoEPSPrescripcion LIKE 'CCF%' 
      THEN struct(TipoIDPaciente, NoIDPaciente)
  END) AS usuarios_ccf,

  -- 4. Usuarios cuyo código de EPS comienza por 'EAS'
  COUNT(DISTINCT CASE 
      WHEN CodigoEPSPrescripcion LIKE 'EAS%' 
      THEN struct(TipoIDPaciente, NoIDPaciente)
  END) AS usuarios_eas,

  -- 5. Usuarios cuyo código de EPS comienza por 'EPS'
  COUNT(DISTINCT CASE 
      WHEN CodigoEPSPrescripcion LIKE 'EPS%' 
      THEN struct(TipoIDPaciente, NoIDPaciente)
  END) AS usuarios_eps,

  -- 6. Usuarios cuyo código de EPS comienza por 'ESS'
  COUNT(DISTINCT CASE 
      WHEN CodigoEPSPrescripcion LIKE 'ESS%' 
      THEN struct(TipoIDPaciente, NoIDPaciente)
  END) AS usuarios_ess

FROM hive_metastore.db_mipres_suministro.dbo_tsum_tx

-- Opcional: puedes incluir un filtro de fechas aquí
-- WHERE FechaRegistro BETWEEN '2024-01-01' AND current_date()

GROUP BY
  mes_registro

ORDER BY
  mes_registro;
