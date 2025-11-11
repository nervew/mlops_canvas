variable "name" { type = string }
variable "environment" { type = string }
variable "location" { type = string }
variable "resource_group_name" { type = string }
variable "tags" { type = map(string) }
variable "network" {
  type = object({
    address_space = list(string)
    subnets = map(object({
      address_prefix = string
      delegated_service = optional(string)
    }))
  })
}
