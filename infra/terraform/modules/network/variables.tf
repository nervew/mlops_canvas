variable "name" { type = string }
variable "location" { type = string }
variable "resource_group_name" { type = string }
variable "address_space" { type = list(string) }
variable "subnets" {
  type = map(object({
    address_prefix    = string
    delegated_service = optional(string)
  }))
}
variable "tags" { type = map(string) }
