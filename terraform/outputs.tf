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

# ---------------------------------------------------------------------
# DAY 23: NETWORKING, IAM & COMPUTE OUTPUTS
# ---------------------------------------------------------------------
output "vpc_id" {
  value       = aws_vpc.mlops_vpc.id
  description = "ID of the provisioned MLOps VPC"
}

output "public_subnet_id" {
  value       = aws_subnet.public_subnet.id
  description = "ID of the public subnet"
}

output "ec2_security_group_id" {
  value       = aws_security_group.ec2_sg.id
  description = "ID of the EC2 security group"
}

output "ec2_iam_role_arn" {
  value       = aws_iam_role.ec2_mlops_role.arn
  description = "ARN of the EC2 IAM role with S3 and ECR access"
}

output "ec2_instance_id" {
  value       = aws_instance.mlops_server.id
  description = "ID of the provisioned EC2 MLOps compute instance"
}

output "ec2_public_ip" {
  value       = aws_instance.mlops_server.public_ip
  description = "Public IPv4 address of the EC2 MLOps compute instance"
}

output "ec2_public_dns" {
  value       = aws_instance.mlops_server.public_dns
  description = "Public DNS hostname of the EC2 MLOps compute instance"
}

output "fastapi_swagger_url" {
  value       = "http://${aws_instance.mlops_server.public_ip}:8000/docs"
  description = "Swagger UI URL for the FastAPI serving endpoint"
}

output "mlflow_tracking_url" {
  value       = "http://${aws_instance.mlops_server.public_ip}:5000"
  description = "MLflow tracking server UI URL"
}

output "ssh_login_command" {
  value       = "ssh -i <your-key.pem> ubuntu@${aws_instance.mlops_server.public_ip}"
  description = "SSH connection string for remote administration"
}

# ---------------------------------------------------------------------
# DAY 25: GITHUB ACTIONS OIDC ROLE OUTPUT
# ---------------------------------------------------------------------
output "github_actions_role_arn" {
  value       = aws_iam_role.github_actions_role.arn
  description = "IAM Role ARN to configure in GitHub Secrets as AWS_ROLE_TO_ASSUME"
}