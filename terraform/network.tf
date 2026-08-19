# terraform/network.tf

# ---------------------------------------------------------------------
# 1. CUSTOM VPC (VIRTUAL PRIVATE CLOUD)
# Dedicated isolated network for MLOps services.
# ---------------------------------------------------------------------
resource "aws_vpc" "mlops_vpc" {
  cidr_block           = var.vpc_cidr
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = {
    Name = "${var.project_name}-vpc-${var.environment}"
  }
}

# ---------------------------------------------------------------------
# 2. INTERNET GATEWAY (IGW)
# Provides direct connectivity between the VPC and the public internet.
# ---------------------------------------------------------------------
resource "aws_internet_gateway" "mlops_igw" {
  vpc_id = aws_vpc.mlops_vpc.id

  tags = {
    Name = "${var.project_name}-igw-${var.environment}"
  }
}

# ---------------------------------------------------------------------
# 3. PUBLIC SUBNET
# Subnet where the EC2 instance resides. Automatically assigns public IP.
# ---------------------------------------------------------------------
resource "aws_subnet" "public_subnet" {
  vpc_id                  = aws_vpc.mlops_vpc.id
  cidr_block              = var.public_subnet_cidr
  map_public_ip_on_launch = true
  availability_zone       = "${var.aws_region}a"

  tags = {
    Name = "${var.project_name}-public-subnet-${var.environment}"
  }
}

# ---------------------------------------------------------------------
# 4. ROUTE TABLE & INTERNET ROUTE
# Directs non-local traffic (0.0.0.0/0) through the Internet Gateway.
# ---------------------------------------------------------------------
resource "aws_route_table" "public_rt" {
  vpc_id = aws_vpc.mlops_vpc.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.mlops_igw.id
  }

  tags = {
    Name = "${var.project_name}-public-rt-${var.environment}"
  }
}

resource "aws_route_table_association" "public_assoc" {
  subnet_id      = aws_subnet.public_subnet.id
  route_table_id = aws_route_table.public_rt.id
}

# ---------------------------------------------------------------------
# 5. SECURITY GROUP (STATEFUL VIRTUAL FIREWALL)
# Inbound: Port 22 (SSH), Port 8000 (FastAPI), Port 5000 (MLflow)
# Outbound: All traffic (to pull packages, Docker images, S3 data)
# ---------------------------------------------------------------------
resource "aws_security_group" "ec2_sg" {
  name        = "${var.project_name}-ec2-sg-${var.environment}"
  description = "Security group for MLOps FastAPI serving and MLflow tracking server"
  vpc_id      = aws_vpc.mlops_vpc.id

  # Ingress: SSH
  ingress {
    description = "SSH access"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = [var.allowed_ssh_cidr]
  }

  # Ingress: FastAPI serving
  ingress {
    description = "FastAPI model inference endpoint"
    from_port   = 8000
    to_port     = 8000
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # Ingress: MLflow UI
  ingress {
    description = "MLflow tracking server UI"
    from_port   = 5000
    to_port     = 5000
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # Egress: All outbound traffic
  egress {
    description = "Allow all outbound traffic"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "${var.project_name}-ec2-sg-${var.environment}"
  }
}
