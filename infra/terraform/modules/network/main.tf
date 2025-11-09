resource "azurerm_virtual_network" "main" {
  name                = "${var.name}-vnet"
  location            = var.location
  resource_group_name = var.resource_group_name
  address_space       = var.address_space
  tags                = var.tags
}

resource "azurerm_subnet" "subnets" {
  for_each             = var.subnets
  name                 = each.key
  resource_group_name  = var.resource_group_name
  virtual_network_name = azurerm_virtual_network.main.name
  address_prefixes     = [each.value.address_prefix]
  delegation {
    name = "delegation"
    service_delegation {
      name = coalesce(each.value.delegated_service, "Microsoft.ContainerService/managedClusters")
      actions = ["Microsoft.Network/virtualNetworks/subnets/action"]
    }
  }
}

resource "azurerm_log_analytics_workspace" "main" {
  name                = "${var.name}-log"
  location            = var.location
  resource_group_name = var.resource_group_name
  sku                 = "PerGB2018"
  retention_in_days   = 30
  tags                = var.tags
}

output "vnet_id" { value = azurerm_virtual_network.main.id }
output "private_subnet_id" { value = azurerm_subnet.subnets["private"].id }
output "aks_subnet_id" { value = azurerm_subnet.subnets["aks"].id }
output "log_analytics_id" { value = azurerm_log_analytics_workspace.main.id }
