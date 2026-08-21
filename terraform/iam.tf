# terraform/iam.tf

# ---------------------------------------------------------------------
# 1. TRUST POLICY (ASSUME ROLE POLICY)
# Tells AWS STS that only the EC2 service is allowed to assume this role.
# ---------------------------------------------------------------------
data "aws_iam_policy_document" "ec2_assume_role" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = ["ec2.amazonaws.com"]
    }
  }
}

# ---------------------------------------------------------------------
# 2. IAM ROLE
# The logical identity that holds our MLOps permissions.
# ---------------------------------------------------------------------
resource "aws_iam_role" "ec2_mlops_role" {
  name               = "${var.project_name}-ec2-role-${var.environment}"
  assume_role_policy = data.aws_iam_policy_document.ec2_assume_role.json

  tags = {
    Name        = "${var.project_name}-ec2-role-${var.environment}"
    Description = "IAM role for MLOps EC2 compute instance"
  }
}

# ---------------------------------------------------------------------
# 3. LEAST-PRIVILEGE PERMISSIONS POLICY
# Explicitly grants access ONLY to:
#   - S3 Artifact Bucket (models, parquet data batches, MLflow runs)
#   - ECR Repositories (docker auth token & image pulling)
# ---------------------------------------------------------------------
data "aws_iam_policy_document" "ec2_permissions" {
  # Permission A: S3 Read & Write for Artifact Bucket
  statement {
    sid    = "S3ArtifactAccess"
    effect = "Allow"
    actions = [
      "s3:GetObject",
      "s3:PutObject",
      "s3:ListBucket",
      "s3:DeleteObject"
    ]
    resources = [
      aws_s3_bucket.artifact_bucket.arn,
      "${aws_s3_bucket.artifact_bucket.arn}/*"
    ]
  }

  # Permission B: ECR Authorization Token (AWS requires '*' resource for login token)
  statement {
    sid       = "ECRAuthToken"
    effect    = "Allow"
    actions   = ["ecr:GetAuthorizationToken"]
    resources = ["*"]
  }

  # Permission C: ECR Image Pulling (Strictly scoped to our provisioned MLOps ECR repos)
  statement {
    sid    = "ECRImagePull"
    effect = "Allow"
    actions = [
      "ecr:BatchCheckLayerAvailability",
      "ecr:GetDownloadUrlForLayer",
      "ecr:BatchGetImage",
      "ecr:DescribeRepositories",
      "ecr:ListImages"
    ]
    resources = [for repo in aws_ecr_repository.ml_repos : repo.arn]
  }
}

resource "aws_iam_policy" "ec2_mlops_policy" {
  name        = "${var.project_name}-ec2-policy-${var.environment}"
  description = "Least privilege IAM policy for NYC Taxi MLOps EC2 instance"
  policy      = data.aws_iam_policy_document.ec2_permissions.json
}

# ---------------------------------------------------------------------
# 4. POLICY ATTACHMENT
# Binds the permissions policy to the IAM role.
# ---------------------------------------------------------------------
resource "aws_iam_role_policy_attachment" "ec2_policy_attach" {
  role       = aws_iam_role.ec2_mlops_role.name
  policy_arn = aws_iam_policy.ec2_mlops_policy.arn
}

# ---------------------------------------------------------------------
# 5. IAM INSTANCE PROFILE
# The AWS bridge/container that passes the IAM role to the EC2 instance.
# ---------------------------------------------------------------------
resource "aws_iam_instance_profile" "ec2_instance_profile" {
  name = "${var.project_name}-ec2-profile-${var.environment}"
  role = aws_iam_role.ec2_mlops_role.name

  tags = {
    Name = "${var.project_name}-ec2-profile-${var.environment}"
  }
}

# ---------------------------------------------------------------------
# 6. GITHUB ACTIONS OIDC IDENTITY PROVIDER & CD ROLE
# Enables secure keyless auth from GitHub Actions to push containers to ECR.
# ---------------------------------------------------------------------
resource "aws_iam_openid_connect_provider" "github_actions" {
  url            = "https://token.actions.githubusercontent.com"
  client_id_list = ["sts.amazonaws.com"]
  thumbprint_list = [
    "6938fd4d98bab03faadb97b34396831e3780aea1",
    "1c58a3a8518e8759bf075b76b750d4f2df264fcd"
  ]

  tags = {
    Name = "${var.project_name}-github-oidc-${var.environment}"
  }
}

data "aws_iam_policy_document" "github_oidc_assume_role" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRoleWithWebIdentity"]

    principals {
      type        = "Federated"
      identifiers = [aws_iam_openid_connect_provider.github_actions.arn]
    }

    condition {
      test     = "StringEquals"
      variable = "token.actions.githubusercontent.com:aud"
      values   = ["sts.amazonaws.com"]
    }

    condition {
      test     = "StringLike"
      variable = "token.actions.githubusercontent.com:sub"
      values = [
        "repo:${var.github_repository}:*",
        "repo:${var.github_repository}:ref:refs/heads/*"
      ]
    }
  }
}

resource "aws_iam_role" "github_actions_role" {
  name               = "${var.project_name}-github-actions-role-${var.environment}"
  assume_role_policy = data.aws_iam_policy_document.github_oidc_assume_role.json

  tags = {
    Name        = "${var.project_name}-github-actions-role-${var.environment}"
    Description = "IAM role for GitHub Actions CI/CD pipeline via OIDC"
  }
}

# Policy allowing GitHub Actions to authenticate and push Docker images to ECR
data "aws_iam_policy_document" "github_actions_ecr_policy_doc" {
  statement {
    sid       = "ECRAuthToken"
    effect    = "Allow"
    actions   = ["ecr:GetAuthorizationToken"]
    resources = ["*"]
  }

  statement {
    sid    = "ECRPushPull"
    effect = "Allow"
    actions = [
      "ecr:BatchCheckLayerAvailability",
      "ecr:GetDownloadUrlForLayer",
      "ecr:BatchGetImage",
      "ecr:PutImage",
      "ecr:InitiateLayerUpload",
      "ecr:UploadLayerPart",
      "ecr:CompleteLayerUpload",
      "ecr:DescribeRepositories",
      "ecr:ListImages"
    ]
    resources = [for repo in aws_ecr_repository.ml_repos : repo.arn]
  }
}

resource "aws_iam_policy" "github_actions_ecr_policy" {
  name        = "${var.project_name}-github-ecr-policy-${var.environment}"
  description = "Allows GitHub Actions to build and push Docker images to ECR"
  policy      = data.aws_iam_policy_document.github_actions_ecr_policy_doc.json
}

resource "aws_iam_role_policy_attachment" "github_actions_attach" {
  role       = aws_iam_role.github_actions_role.name
  policy_arn = aws_iam_policy.github_actions_ecr_policy.arn
}
