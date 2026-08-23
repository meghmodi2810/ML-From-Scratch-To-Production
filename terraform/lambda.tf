# terraform/lambda.tf

# ---------------------------------------------------------------------
# 1. IAM ROLE FOR AWS LAMBDA (SERVERLESS INFERENCE)
# ---------------------------------------------------------------------
data "aws_iam_policy_document" "lambda_assume_role" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = ["lambda.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "lambda_exec_role" {
  name               = "${var.project_name}-lambda-role-${var.environment}"
  assume_role_policy = data.aws_iam_policy_document.lambda_assume_role.json

  tags = {
    Name        = "${var.project_name}-lambda-role-${var.environment}"
    Description = "IAM Role for Serverless ML Inference Lambda"
  }
}

# Attach AWS managed basic execution role for CloudWatch Logs
resource "aws_iam_role_policy_attachment" "lambda_basic_logs" {
  role       = aws_iam_role.lambda_exec_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

# Attach S3 read policy for model weights
data "aws_iam_policy_document" "lambda_s3_policy_doc" {
  statement {
    sid    = "S3ArtifactRead"
    effect = "Allow"
    actions = [
      "s3:GetObject",
      "s3:ListBucket"
    ]
    resources = [
      aws_s3_bucket.artifact_bucket.arn,
      "${aws_s3_bucket.artifact_bucket.arn}/*"
    ]
  }
}

resource "aws_iam_policy" "lambda_s3_policy" {
  name        = "${var.project_name}-lambda-s3-policy-${var.environment}"
  description = "Allows Lambda to read MLflow models from S3 artifact bucket"
  policy      = data.aws_iam_policy_document.lambda_s3_policy_doc.json
}

resource "aws_iam_role_policy_attachment" "lambda_s3_attach" {
  role       = aws_iam_role.lambda_exec_role.name
  policy_arn = aws_iam_policy.lambda_s3_policy.arn
}

# ---------------------------------------------------------------------
# 2. AWS LAMBDA FUNCTION (CONTAINER IMAGE PACKAGE TYPE)
# ---------------------------------------------------------------------
resource "aws_lambda_function" "mlops_lambda" {
  function_name = "${var.project_name}-serverless-inference-${var.environment}"
  role          = aws_iam_role.lambda_exec_role.arn
  package_type  = "Image"
  image_uri     = "${aws_ecr_repository.ml_repos["nyc-taxi-lambda"].repository_url}:latest"
  timeout       = 30
  memory_size   = 1024 # 1 GB RAM for fast Cold Starts & sub-20ms inference

  environment {
    variables = {
      ENVIRONMENT         = var.environment
      DATA_DIR            = "/tmp"
      MLFLOW_TRACKING_URI = "sqlite:////tmp/mlflow.db"
    }
  }

  tags = {
    Name        = "${var.project_name}-serverless-inference-${var.environment}"
    Environment = var.environment
  }
}

# ---------------------------------------------------------------------
# 3. AMAZON API GATEWAY (HTTP API V2)
# ---------------------------------------------------------------------
resource "aws_apigatewayv2_api" "http_api" {
  name          = "${var.project_name}-serverless-api-${var.environment}"
  protocol_type = "HTTP"
  description   = "Serverless HTTP API trigger for NYC Taxi ML Tip Prediction"

  cors_configuration {
    allow_origins = ["*"]
    allow_methods = ["GET", "POST", "OPTIONS"]
    allow_headers = ["Content-Type", "Authorization"]
    max_age       = 300
  }

  tags = {
    Name        = "${var.project_name}-api-${var.environment}"
    Environment = var.environment
  }
}

# API Gateway Integration to Lambda
resource "aws_apigatewayv2_integration" "lambda_integration" {
  api_id                 = aws_apigatewayv2_api.http_api.id
  integration_type       = "AWS_PROXY"
  integration_uri        = aws_lambda_function.mlops_lambda.invoke_arn
  integration_method     = "POST"
  payload_format_version = "2.0"
}

# API Gateway Routes: /predict and /health
resource "aws_apigatewayv2_route" "predict_route" {
  api_id    = aws_apigatewayv2_api.http_api.id
  route_key = "POST /predict"
  target    = "integrations/${aws_apigatewayv2_integration.lambda_integration.id}"
}

resource "aws_apigatewayv2_route" "health_route" {
  api_id    = aws_apigatewayv2_api.http_api.id
  route_key = "GET /health"
  target    = "integrations/${aws_apigatewayv2_integration.lambda_integration.id}"
}

resource "aws_apigatewayv2_route" "default_route" {
  api_id    = aws_apigatewayv2_api.http_api.id
  route_key = "$default"
  target    = "integrations/${aws_apigatewayv2_integration.lambda_integration.id}"
}

# $default Stage (Auto-deployed)
resource "aws_apigatewayv2_stage" "default_stage" {
  api_id      = aws_apigatewayv2_api.http_api.id
  name        = "$default"
  auto_deploy = true

  tags = {
    Name = "${var.project_name}-default-stage-${var.environment}"
  }
}

# ---------------------------------------------------------------------
# 4. LAMBDA INVOKE PERMISSION FOR API GATEWAY
# ---------------------------------------------------------------------
resource "aws_lambda_permission" "api_gateway_permission" {
  statement_id  = "AllowExecutionFromAPIGateway"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.mlops_lambda.function_name
  principal     = "apigateway.amazonaws.com"
  source_arn    = "${aws_apigatewayv2_api.http_api.execution_arn}/*/*"
}
