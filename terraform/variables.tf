# terraform/variables.tf
variable "aws_region" {
  type        = string
  description = "AWS region for provisioning resources"
  default     = "ap-south-1"
}

variable "environment" {
  type        = string
  description = "Deployment environment (e.g., dev, staging, prod)"
  default     = "dev"
}

variable "project_name" {
  type        = string
  description = "Project prefix for resource naming"
  default     = "nyc-taxi-mlops"
}

variable "s3_bucket_prefix" {
  type        = string
  description = "Prefix for S3 artifact bucket"
  default     = "nyc-taxi-artifacts"
}

variable "ecr_repository_names" {
  type        = list(string)
  description = "List of ECR repositories to provision for ML training and serving"
  default     = ["nyc-taxi-train", "nyc-taxi-serve"]
}