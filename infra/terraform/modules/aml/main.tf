resource "azurerm_application_insights" "main" {
  name                = "${var.name}-appi"
  location            = var.location
  resource_group_name = var.resource_group_name
  application_type    = "web"
  workspace_id        = var.log_analytics_id
  tags                = var.tags
}

resource "azurerm_storage_account" "main" {
  name                     = replace(lower("${var.name}aml"), "-", "")
  resource_group_name      = var.resource_group_name
  location                 = var.location
  account_tier             = "Standard"
  account_replication_type = "LRS"
  enable_https_traffic_only = true
  min_tls_version          = "TLS1_2"
  large_file_share_enabled = false
  tags                     = var.tags
}

resource "azurerm_key_vault" "main" {
  name                        = "${var.name}-kv"
  location                    = var.location
  resource_group_name         = var.resource_group_name
  tenant_id                   = data.azurerm_client_config.current.tenant_id
  sku_name                    = "standard"
  purge_protection_enabled    = true
  soft_delete_retention_days  = 7
  tags                        = var.tags
}

data "azurerm_client_config" "current" {}

resource "azurerm_container_registry" "main" {
  name                     = replace(upper("${var.name}acr"), "-", "")
  resource_group_name      = var.resource_group_name
  location                 = var.location
  sku                      = "Premium"
  admin_enabled            = false
  public_network_access_enabled = false
  tags                     = var.tags
}

resource "azurerm_kubernetes_cluster" "aks" {
  name                = "${var.name}-aks"
  location            = var.location
  resource_group_name = var.resource_group_name
  dns_prefix          = "${var.name}-aks"
  private_cluster_enabled = true
  default_node_pool {
    name       = "system"
    node_count = 3
    vm_size    = "Standard_D4s_v5"
    vnet_subnet_id = var.aks_subnet_id
  }
  identity {
    type = "SystemAssigned"
  }
  tags = var.tags
}

resource "azurerm_machine_learning_workspace" "main" {
  name                = "${var.name}-aml"
  location            = var.location
  resource_group_name = var.resource_group_name
  application_insights_id = azurerm_application_insights.main.id
  key_vault_id            = azurerm_key_vault.main.id
  storage_account_id      = azurerm_storage_account.main.id
  container_registry_id   = azurerm_container_registry.main.id
  public_network_access_enabled = false
  identity {
    type = "SystemAssigned"
  }
  primary_user_assigned_identity_id = null
  tags = var.tags
}

output "workspace_id" { value = azurerm_machine_learning_workspace.main.id }
output "key_vault_id" { value = azurerm_key_vault.main.id }
output "container_registry_login_server" { value = azurerm_container_registry.main.login_server }
