# terraform/outputs.tf
output "s3_artifact_bucket_name" {
  value       = aws_s3_bucket.artifact_bucket.id
  description = "Name of the provisioned S3 artifact bucket"
}

output "s3_artifact_bucket_arn" {
  value       = aws_s3_bucket.artifact_bucket.arn
  description = "ARN of the provisioned S3 artifact bucket"
}

output "dynamodb_lock_table_name" {
  value       = aws_dynamodb_table.terraform_locks.name
  description = "Name of the DynamoDB table used for state locking"
}

output "ecr_repository_urls" {
  value = {
    for k, v in aws_ecr_repository.ml_repos : k => v.repository_url
  }
  description = "Registry URLs for provisioned ECR container repositories"
}