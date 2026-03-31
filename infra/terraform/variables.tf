# Person B — Terraform Variables (Weeks 3-4)

variable "aws_region" {
  description = "AWS region to deploy into"
  type        = string
  default     = "us-east-1"
}

variable "project_name" {
  description = "Prefix for all AWS resource names"
  type        = string
  default     = "capstone"
}

variable "environment" {
  description = "Environment tag"
  type        = string
  default     = "demo"
}

# EKS
variable "eks_node_instance_type" {
  description = "EC2 instance type for EKS worker nodes (keep small for budget)"
  type        = string
  default     = "t3.small"  # ~$0.021/hr each, 2 nodes = ~$0.042/hr
}

variable "eks_desired_nodes" {
  type    = number
  default = 2
}

# MSK (Kafka)
variable "msk_instance_type" {
  description = "MSK broker instance type"
  type        = string
  default     = "kafka.t3.small"  # ~$0.082/hr
}

# ECR (one repo per service)
variable "ecr_repos" {
  description = "List of ECR repository names to create"
  type        = list(string)
  default     = ["embedding", "rag", "reward-model", "agents", "kafka-consumer", "spring-boot-api", "frontend"]
}
