variable "prefix" {
  description = "Prefijo para los recursos."
  type        = string
  default     = "mlopstest"
}

variable "resource_group_name" {
  description = "Nombre del resource group existente donde desplegar."
  type        = string
  default     = "GRPANALITICA"
}

variable "location" {
  description = "Ubicación de Azure para los recursos."
  type        = string
  default     = "eastus"
}

variable "container_image" {
  description = "Imagen completa del contenedor a desplegar. Si se omite se usa la del ACR creado."
  type        = string
  default     = null
}
