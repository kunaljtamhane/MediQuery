# Person B — Terraform Root Module (Weeks 3-4: write/plan, Weeks 9-10: apply)
#
# IMPORTANT: Do NOT run `terraform apply` until Week 9.
#            Run `terraform plan` freely — it costs nothing.
#            Run `terraform destroy` every night once deployed.
#
# Deploy:   terraform apply -var-file=terraform.tfvars
# Destroy:  terraform destroy -var-file=terraform.tfvars  ← RUN THIS EVERY NIGHT

terraform {
  required_version = ">= 1.8"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  # State stored in S3 with DynamoDB locking
  # TODO Week 9: Create the S3 bucket and DynamoDB table manually first, then uncomment:
  # backend "s3" {
  #   bucket         = "capstone-terraform-state"
  #   key            = "capstone/terraform.tfstate"
  #   region         = "us-east-1"
  #   dynamodb_table = "capstone-terraform-locks"
  #   encrypt        = true
  # }
}

provider "aws" {
  region = var.aws_region
  default_tags {
    tags = {
      Project     = var.project_name
      Environment = var.environment
      ManagedBy   = "Terraform"
    }
  }
}

# ── VPC ───────────────────────────────────────────────────────────────────────
module "vpc" {
  source       = "./modules/vpc"
  project_name = var.project_name
}

# ── EKS ───────────────────────────────────────────────────────────────────────
module "eks" {
  source              = "./modules/eks"
  project_name        = var.project_name
  vpc_id              = module.vpc.vpc_id
  subnet_ids          = module.vpc.private_subnet_ids
  node_instance_type  = var.eks_node_instance_type
  desired_nodes       = var.eks_desired_nodes
}

# ── MSK (Kafka) ───────────────────────────────────────────────────────────────
module "msk" {
  source         = "./modules/msk"
  project_name   = var.project_name
  vpc_id         = module.vpc.vpc_id
  subnet_ids     = module.vpc.private_subnet_ids
  instance_type  = var.msk_instance_type
}

# ── S3 (document storage / model artifacts) ───────────────────────────────────
resource "aws_s3_bucket" "artifacts" {
  bucket = "${var.project_name}-artifacts-${data.aws_caller_identity.current.account_id}"
  force_destroy = true  # OK for capstone — don't use in production
}

resource "aws_s3_bucket_versioning" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id
  versioning_configuration { status = "Enabled" }
}

# ── ECR (one repo per Docker image) ──────────────────────────────────────────
resource "aws_ecr_repository" "services" {
  for_each             = toset(var.ecr_repos)
  name                 = "${var.project_name}/${each.value}"
  image_tag_mutability = "MUTABLE"
  force_delete         = true

  image_scanning_configuration { scan_on_push = true }
}

data "aws_caller_identity" "current" {}
