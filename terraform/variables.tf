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

# ---------------------------------------------------------------------
# NETWORKING VARIABLES
# ---------------------------------------------------------------------
variable "vpc_cidr" {
  type        = string
  description = "CIDR block for the custom MLOps VPC"
  default     = "10.0.0.0/16"
}

variable "public_subnet_cidr" {
  type        = string
  description = "CIDR block for the public subnet"
  default     = "10.0.1.0/24"
}

variable "allowed_ssh_cidr" {
  type        = string
  description = "CIDR block allowed to SSH into the EC2 instance"
  default     = "0.0.0.0/0"
}

# ---------------------------------------------------------------------
# COMPUTE VARIABLES
# ---------------------------------------------------------------------
variable "instance_type" {
  type        = string
  description = "EC2 instance type for model training and serving"
  default     = "t3.small"
}

variable "key_name" {
  type        = string
  description = "Name of an existing AWS SSH Key Pair (leave blank if not using SSH key)"
  default     = ""
}

# ---------------------------------------------------------------------
# CI/CD & OIDC VARIABLES
# ---------------------------------------------------------------------
variable "github_repository" {
  type        = string
  description = "GitHub repository name (owner/repo) authorized to assume ECR push role"
  default     = "meghmodi2810/ML-From-Scratch-To-Production"
}