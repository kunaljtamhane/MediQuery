# Person B — Terraform Outputs
# Run `terraform output` after apply to get connection details

output "eks_cluster_name" {
  description = "EKS cluster name — use with: aws eks update-kubeconfig --name <value>"
  value       = module.eks.cluster_name
}

output "eks_cluster_endpoint" {
  value = module.eks.cluster_endpoint
}

output "msk_bootstrap_brokers" {
  description = "MSK Kafka bootstrap server string for application.properties"
  value       = module.msk.bootstrap_brokers
}

output "s3_bucket_name" {
  value = aws_s3_bucket.artifacts.bucket
}

output "ecr_registry" {
  description = "ECR registry URL — prefix all image names with this"
  value       = "${data.aws_caller_identity.current.account_id}.dkr.ecr.${var.aws_region}.amazonaws.com"
}
