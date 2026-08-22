# terraform/compute.tf

# ---------------------------------------------------------------------
# 1. DYNAMIC AMI LOOKUP
# Automatically fetches the latest official Canonical Ubuntu 22.04 LTS AMI.
# ---------------------------------------------------------------------
data "aws_ami" "ubuntu" {
  most_recent = true
  owners      = ["099720109477"] # Canonical official owner ID

  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}

# ---------------------------------------------------------------------
# 2. EC2 MLOPS COMPUTE INSTANCE
# Dedicated compute server with Docker runtime & IAM instance profile.
# ---------------------------------------------------------------------
resource "aws_instance" "mlops_server" {
  ami                    = data.aws_ami.ubuntu.id
  instance_type          = var.instance_type
  subnet_id              = aws_subnet.public_subnet.id
  vpc_security_group_ids = [aws_security_group.ec2_sg.id]
  iam_instance_profile   = aws_iam_instance_profile.ec2_instance_profile.name
  key_name               = var.key_name != "" ? var.key_name : null

  # Root EBS Volume (20GB gp3 SSD)
  root_block_device {
    volume_size           = 20
    volume_type           = "gp3"
    delete_on_termination = true
    encrypted             = true

    tags = {
      Name = "${var.project_name}-root-volume-${var.environment}"
    }
  }

  # ---------------------------------------------------------------------
  # 3. USER DATA BOOTSTRAP SCRIPT
  # Automated cloud-init startup script executed as root on launch:
  # - Updates packages
  # - Installs Docker Engine & Docker Compose Plugin
  # - Starts Docker daemon
  # - Grants ubuntu user non-root Docker privileges
  # - Installs AWS CLI v2
  # ---------------------------------------------------------------------
  user_data = <<-EOF
              #!/bin/bash
              set -e

              # 1. Update OS packages
              apt-get update -y
              apt-get install -y apt-transport-https ca-certificates curl software-properties-common unzip gnupg lsb-release

              # 2. Install Docker Engine
              install -m 0755 -d /etc/apt/keyrings
              curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
              chmod a+r /etc/apt/keyrings/docker.gpg

              echo \
                "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
                $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null

              apt-get update -y
              apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

              # 3. Enable and start Docker service
              systemctl enable docker
              systemctl start docker

              # 4. Add ubuntu user to docker group
              usermod -aG docker ubuntu

              # 5. Install AWS CLI v2
              curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
              unzip -q awscliv2.zip
              ./aws/install
              rm -rf aws awscliv2.zip

              # 6. Ensure Amazon SSM Agent is active for zero-touch deployments
              snap install amazon-ssm-agent --classic || true
              systemctl enable snap.amazon-ssm-agent.amazon-ssm-agent.service || true
              systemctl restart snap.amazon-ssm-agent.amazon-ssm-agent.service || true

              # 7. Mark bootstrapping as complete
              echo "NYC Taxi MLOps Server Bootstrap Completed at $(date)" > /var/log/user_data_complete.log
              EOF

  user_data_replace_on_change = true

  tags = {
    Name = "${var.project_name}-ec2-${var.environment}"
  }
}
