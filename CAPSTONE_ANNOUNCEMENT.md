# 📢 Capstone Launch & Portfolio Recap: 30 Days of ML

> **Ready-to-publish LinkedIn / Twitter Post for Capstone Release v1.0.0**

---

### 🚀 LinkedIn Post (Copy-Paste Ready)

```text
🚀 I just completed a 30-Day Engineering Challenge: "Machine Learning: From Scratch to Production"!

Most ML tutorials end at `model.fit()` inside a Jupyter Notebook. But in the real world, the true challenge is understanding the deep mathematical mechanics of algorithms from first principles and packaging them into reliable, automated, monitored production systems in the cloud.

Over the past 30 days, I took on both sides of that equation:

🔬 Phase 1: Machine Learning Foundations (Days 1–13)
Instead of importing scikit-learn, I hand-derived and implemented core algorithms in pure NumPy:
• Linear & Logistic Regression with closed-form Normal Equations and SGD
• CART Decision Trees with Gini Impurity & Recursive Splitting
• Random Forests with Bootstrap Aggregation and Feature Subsampling
• Adaptive Boosting (AdaBoost) with sample weight updates
• XGBoost from Scratch using 2nd-order Taylor series expansions (gradients & hessians)
• Support Vector Machines (SVM) with Sequential Minimal Optimization (SMO)
• Gaussian Naive Bayes & K-Nearest Neighbors

⚙️ Phase 2: Enterprise MLOps & Production Engineering (Days 14–30)
I transitioned to the real-world NYC TLC Yellow Taxi dataset to experience real production friction and built a production cloud architecture on AWS:
• Data Contracts & Validation: Enforced schema rules, types, and boundaries using Pandera.
• Experiment Tracking & Model Governance: Managed model lifecycle stages (Staging ➔ Production) with MLflow.
• High-Performance Serving API: Built a FastAPI inference service with Pydantic v2 schemas and threadpool offloading.
• Infrastructure as Code: Provisioned AWS VPC, EC2, ECR, IAM, and Security Groups declaratively with Terraform.
• Automated CI/CD/CT: Automated linting, 3-tier testing gates, OIDC AWS authentication, and zero-downtime rolling deployments with GitHub Actions.
• Serverless Architecture: Deployed containerized serverless inference on AWS Lambda & API Gateway.
• Real-time Observability: Emitted Prometheus telemetry and visualized live p50/p95/p99 latency and feature drift on Grafana.
• Load Testing & SLAs: Benchmarked concurrent traffic using Locust, achieving sub-40ms median latency and 0% failure rate under nominal load.

💡 Key Engineering Takeaway:
Running CPU-bound synchronous ML inference inside FastAPI's `async def` blocks the single asyncio event loop thread, creating head-of-line blocking under load. Switching to synchronous `def` routes allows Starlette's threadpool to process predictions concurrently, dropping p95 latency by over 50%!

Check out the full repository, architecture diagrams, and mathematical proofs here:
🔗 GitHub: https://github.com/meghmodi2810/ML-From-Scratch-To-Production

A huge thank you to everyone following along with the journey! 🚀

#MachineLearning #MLOps #Python #AWS #FastAPI #Docker #Terraform #DevOps #DataScience #DataEngineering #AI #SoftwareEngineering #XGBoost #Prometheus #Grafana
```

---

### 🐦 Twitter / X Thread Version

```text
🧵 1/5 🚀 I just completed the 30-Day "ML From Scratch to Production" challenge!

From deriving algorithms in pure NumPy to building a production MLOps stack on AWS with Terraform, Docker, MLflow, Prometheus, & Locust.

Here’s the complete breakdown 👇

2/5 🔬 Phase 1: Pure NumPy Algorithms
Wrote ML algorithms from first principles:
• Linear/Logistic Regression (SGD & Closed-form)
• CART Decision Trees & Random Forests
• AdaBoost & XGBoost (2nd-order Taylor expansions)
• SVM with Sequential Minimal Optimization (SMO)

3/5 ⚙️ Phase 2: Enterprise MLOps Stack
• Data contracts with Pandera
• MLflow tracking & model registry
• FastAPI serving with threadpool concurrency
• Terraform IaC for AWS VPC, EC2, IAM, & Lambda
• 3-tier CI/CD/CT via GitHub Actions & OIDC

4/5 📊 Observability & Load Testing
• Prometheus metrics (`http_requests_total`, latency histograms)
• Live Grafana dashboards
• Locust load testing: 1,400+ requests, 39ms median latency, 0% error rate.

5/5 🔗 Check out the open-source repo, architecture diagrams, & math notes:
https://github.com/meghmodi2810/ML-From-Scratch-To-Production

#MLOps #MachineLearning #Python #AWS
```
