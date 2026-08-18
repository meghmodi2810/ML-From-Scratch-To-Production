# terraform/versions.tf
terraform {
  required_version = ">= 1.5.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.0"
    }
  }

  # NOTE: To enable remote state after creating backend resources, uncomment this block:
  backend "s3" {
    bucket       = "nyc-taxi-mlops-tf-state-2026"
    key          = "mlops/dev/terraform.tfstate"
    region       = "ap-south-1"
    use_lockfile = true
    encrypt      = true
  }
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project     = "NYC-Taxi-MLOps"
      Environment = var.environment
      ManagedBy   = "Terraform"
      Day         = "Day22"
    }
  }
}