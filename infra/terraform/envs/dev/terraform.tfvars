name                = "mlops-dev"
environment         = "dev"
location            = "eastus2"
resource_group_name = "rg-mlops-dev"
tags = {
  environment = "dev"
  project     = "mlops-canvas"
}
network = {
  address_space = ["10.10.0.0/16"]
  subnets = {
    private = {
      address_prefix = "10.10.1.0/24"
    }
    aks = {
      address_prefix = "10.10.2.0/24"
    }
  }
}
