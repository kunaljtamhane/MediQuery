# Person B — MSK (Managed Kafka) Module
# Single broker, single AZ — minimal for capstone budget (~$0.082/hr)

variable "project_name"  { type = string }
variable "vpc_id"        { type = string }
variable "subnet_ids"    { type = list(string) }
variable "instance_type" { type = string }

resource "aws_security_group" "msk" {
  name   = "${var.project_name}-msk-sg"
  vpc_id = var.vpc_id

  ingress {
    description = "Kafka plaintext from within VPC"
    from_port   = 9092
    to_port     = 9092
    protocol    = "tcp"
    cidr_blocks = ["10.0.0.0/16"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_msk_cluster" "kafka" {
  cluster_name           = "${var.project_name}-kafka"
  kafka_version          = "3.6.0"
  number_of_broker_nodes = 1

  broker_node_group_info {
    instance_type  = var.instance_type
    client_subnets = [var.subnet_ids[0]]  # single AZ to minimise cost
    storage_info {
      ebs_storage_info { volume_size = 10 }
    }
    security_groups = [aws_security_group.msk.id]
  }

  encryption_info {
    encryption_in_transit { client_broker = "PLAINTEXT" }
  }
}

output "bootstrap_brokers" {
  value = aws_msk_cluster.kafka.bootstrap_brokers
}
