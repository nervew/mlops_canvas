data "azurerm_resource_group" "rg" {
  name = var.resource_group_name
}

resource "random_id" "suffix" {
  byte_length = 2
  keepers = {
    prefix = var.prefix
  }
}

locals {
  acr_name          = lower("${var.prefix}${random_id.suffix.hex}acr")
  workspace_name    = "${var.prefix}-law"
  containerapp_env  = "${var.prefix}-aca-env"
  containerapp_name = "${var.prefix}-api"
  default_image     = "${lower(local.acr_name)}.azurecr.io/${var.prefix}-api:latest"
}

resource "azurerm_log_analytics_workspace" "logs" {
  name                = local.workspace_name
  location            = data.azurerm_resource_group.rg.location
  resource_group_name = data.azurerm_resource_group.rg.name
  sku                 = "PerGB2018"
  retention_in_days   = 30
}

resource "azurerm_container_registry" "acr" {
  name                = local.acr_name
  resource_group_name = data.azurerm_resource_group.rg.name
  location            = data.azurerm_resource_group.rg.location
  sku                 = "Standard"
  admin_enabled       = false
}

resource "azurerm_user_assigned_identity" "acr_identity" {
  name                = "${var.prefix}-aca-identity"
  resource_group_name = data.azurerm_resource_group.rg.name
  location            = data.azurerm_resource_group.rg.location
}

resource "azurerm_role_assignment" "acr_pull" {
  scope                = azurerm_container_registry.acr.id
  role_definition_name = "AcrPull"
  principal_id         = azurerm_user_assigned_identity.acr_identity.principal_id
}

resource "azurerm_container_app_environment" "env" {
  name                       = local.containerapp_env
  location                   = data.azurerm_resource_group.rg.location
  resource_group_name        = data.azurerm_resource_group.rg.name
  log_analytics_workspace_id = azurerm_log_analytics_workspace.logs.id
}

resource "azurerm_container_app" "app" {
  name                         = local.containerapp_name
  resource_group_name          = data.azurerm_resource_group.rg.name
  container_app_environment_id = azurerm_container_app_environment.env.id

  revision_mode = "Single"

  identity {
    type = "UserAssigned"
    user_assigned_identities = {
      (azurerm_user_assigned_identity.acr_identity.id) = {}
    }
  }

  registry {
    server = azurerm_container_registry.acr.login_server
    identity {
      type        = "UserAssigned"
      resource_id = azurerm_user_assigned_identity.acr_identity.id
    }
  }

  ingress {
    external_enabled = true
    target_port      = 8080
    transport        = "auto"
    traffic_weight {
      latest_revision = true
      percentage      = 100
    }
  }

  template {
    container {
      name   = "${var.prefix}-api"
      image  = coalesce(var.container_image, local.default_image)
      cpu    = 0.25
      memory = "0.5Gi"
      env {
        name  = "PORT"
        value = "8080"
      }
    }
  }
}

output "container_app_fqdn" {
  description = "URL pública de la Container App."
  value       = azurerm_container_app.app.latest_revision_fqdn
}

output "acr_login_server" {
  description = "Login server del registro ACR."
  value       = azurerm_container_registry.acr.login_server
}

output "container_app_name" {
  description = "Nombre de la Container App."
  value       = azurerm_container_app.app.name
}
