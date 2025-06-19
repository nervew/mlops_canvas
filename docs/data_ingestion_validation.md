# Data Ingestion Validation Module

Este modulo implementa validaciones de datos de ingesta siguiendo arquitectura hexagonal. Asegura que los datos cumplen requisitos basicos antes de entrenar modelos.

```mermaid
flowchart TD
    A[Load config] --> B[Select Validator]
    B --> C[Validate Data]
    C --> D[Generate Report]
    D --> E[Upload to Databricks]
```
