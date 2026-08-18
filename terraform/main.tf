# terraform/main.tf

# ---------------------------------------------------------------------
# 1. S3 ARTIFACT BUCKET (MLflow Artifacts & Parquet Batches)
# ---------------------------------------------------------------------
resource "random_id" "bucket_suffix" {
  byte_length = 4
}

resource "aws_s3_bucket" "artifact_bucket" {
  bucket        = "${var.s3_bucket_prefix}-${var.environment}-${random_id.bucket_suffix.hex}"
  force_destroy = true # Allows clean tear down in sandbox environments
}

# Enable S3 Bucket Versioning for artifact lineage
resource "aws_s3_bucket_versioning" "artifact_bucket_versioning" {
  bucket = aws_s3_bucket.artifact_bucket.id
  versioning_configuration {
    status = "Enabled"
  }
}

# Enable Server-Side Encryption (AES256)
resource "aws_s3_bucket_server_side_encryption_configuration" "artifact_bucket_encryption" {
  bucket = aws_s3_bucket.artifact_bucket.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# Block all Public Access (Security Compliance)
resource "aws_s3_bucket_public_access_block" "artifact_bucket_public_block" {
  bucket = aws_s3_bucket.artifact_bucket.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# ---------------------------------------------------------------------
# 2. DYNAMODB TABLE FOR TERRAFORM STATE LOCKING
# ---------------------------------------------------------------------
resource "aws_dynamodb_table" "terraform_locks" {
  name         = "${var.project_name}-tf-locks-${var.environment}"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "LockID"

  attribute {
    name = "LockID"
    type = "S"
  }
}

# ---------------------------------------------------------------------
# 3. AWS ECR REPOSITORIES (TRAIN & SERVE CONTAINERS)
# ---------------------------------------------------------------------
resource "aws_ecr_repository" "ml_repos" {
  for_each             = toset(var.ecr_repository_names)
  name                 = "${var.project_name}-${each.key}-${var.environment}"
  image_tag_mutability = "MUTABLE"

  # Automatic vulnerability scan on push
  image_scanning_configuration {
    scan_on_push = true
  }

  encryption_configuration {
    encryption_type = "AES256"
  }
}

# ECR Lifecycle Policy (Keep last 10 images to save storage costs)
resource "aws_ecr_lifecycle_policy" "ml_repos_policy" {
  for_each   = aws_ecr_repository.ml_repos
  repository = each.value.name

  policy = jsonencode({
    rules = [
      {
        rulePriority = 1
        description  = "Keep last 10 images"
        selection = {
          tagStatus   = "any"
          countType   = "imageCountMoreThan"
          countNumber = 10
        }
        action = {
          type = "expire"
        }
      }
    ]
  })
}