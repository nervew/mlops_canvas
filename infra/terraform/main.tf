terraform {
  required_version = ">= 1.4.0"
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.70"
    }
  }
  backend "azurerm" {}
}

provider "azurerm" {
  features {}
}

locals {
  environment = var.environment
}

module "network" {
  source              = "./modules/network"
  name                = var.name
  location            = var.location
  resource_group_name = azurerm_resource_group.main.name
  address_space       = var.network.address_space
  subnets             = var.network.subnets
}

resource "azurerm_resource_group" "main" {
  name     = var.resource_group_name
  location = var.location
  tags     = var.tags
}

module "aml" {
  source              = "./modules/aml"
  name                = var.name
  location            = var.location
  resource_group_name = azurerm_resource_group.main.name
  tags                = var.tags
  vnet_id             = module.network.vnet_id
  subnet_id           = module.network.private_subnet_id
  log_analytics_id    = module.network.log_analytics_id
  aks_subnet_id       = module.network.aks_subnet_id
}
