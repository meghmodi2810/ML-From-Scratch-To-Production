# 📖 Technical Glossary & Engineering Lexicon

A comprehensive reference dictionary of all mathematical, algorithmic, cloud, and MLOps concepts mastered during the **30 Days of ML: From Scratch to Production** challenge.

---

## 📑 Table of Contents
1. [Mathematical Foundations & Optimization](#1-mathematical-foundations--optimization)
2. [Core Machine Learning & Ensemble Algorithms](#2-core-machine-learning--ensemble-algorithms)
3. [Data Engineering, Validation & Contracts](#3-data-engineering-validation--contracts)
4. [Experiment Tracking, Governance & Metadata](#4-experiment-tracking-governance--metadata)
5. [Cloud Infrastructure, Networking & Containers](#5-cloud-infrastructure-networking--containers)
6. [Automated CI/CD/CT Pipelines & Quality Gates](#6-automated-cicdct-pipelines--quality-gates)
7. [Observability, Telemetry & Load Testing](#7-observability-telemetry--load-testing)

---

## 1. Mathematical Foundations & Optimization

### Gradient Descent (GD)
An iterative first-order optimization algorithm for finding a local minimum of a differentiable function. The algorithm computes the partial derivative (gradient $\nabla_\theta J(\theta)$) of the loss function with respect to model parameters and steps in the opposite direction scaled by a learning rate $\eta$:
$$\theta_{t+1} = \theta_t - \eta \nabla_\theta J(\theta_t)$$

### Stochastic Gradient Descent (SGD)
A variant of Gradient Descent that updates parameters using a single training example (or small mini-batch) per iteration rather than the entire dataset. This introduces stochastic noise that helps escape shallow local minima while drastically decreasing memory overhead and computation time per step.

### Normal Equation (Closed-Form Solution)
The analytical matrix solution for Ordinary Least Squares (OLS) Linear Regression that directly minimizes the Mean Squared Error without iteration:
$$\theta = (X^T X)^{-1} X^T y$$
Guaranteed to find the exact global minimum when $(X^T X)$ is non-singular (invertible).

### Convex Optimization
A mathematical subfield dealing with functions where every local minimum is also the global minimum. Characterized by a positive semi-definite Hessian matrix ($\nabla^2 f(x) \succeq 0$). MSE loss and Binary Cross-Entropy loss for linear models are convex.

### $L_1$ Regularization (Lasso)
Adds the sum of absolute values of weights as a penalty term to the loss function ($\lambda \sum |\theta_j|$). Drives non-essential feature weights strictly to zero, performing automatic feature selection and creating sparse models.

### $L_2$ Regularization (Ridge / Tikhonov)
Adds the sum of squared weights as a penalty term ($\frac{\lambda}{2} \sum \theta_j^2$). Penalizes large weight magnitudes smoothly, preventing multicollinearity and reducing model variance without forcing weights to absolute zero.

### 2nd-Order Taylor Series Expansion
A polynomial approximation of a differentiable function around a point using first derivatives (gradients) and second derivatives (hessians):
$$f(x + \Delta x) \approx f(x) + f'(x)\Delta x + \frac{1}{2}f''(x)\Delta x^2$$
Formulates the exact loss optimization mechanics in algorithms like Newton-Raphson and XGBoost.

---

## 2. Core Machine Learning & Ensemble Algorithms

### Gini Impurity
A measure of statistical dispersion and node impurity used in the CART (Classification and Regression Trees) decision tree algorithm:
$$\text{Gini}(D) = 1 - \sum_{k=1}^K p_k^2$$
A pure node containing only one class has a Gini impurity of $0.0$.

### Information Gain & Entropy
Information Entropy measures the expected uncertainty or information content in a random variable:
$$H(D) = -\sum_{k=1}^K p_k \log_2(p_k)$$
Information Gain is the reduction in entropy achieved by partitioning a dataset on a specific attribute: $\text{Gain}(D, A) = H(D) - \sum \frac{|D_v|}{|D|} H(D_v)$.

### Bootstrap Aggregation (Bagging)
An ensemble meta-algorithm where multiple base learners (e.g. Decision Trees) are trained independently in parallel on bootstrap samples (random sampling with replacement). Predictions are aggregated via majority voting (classification) or averaging (regression), drastically reducing model variance.

### Random Forests
An extension of Bagging where, in addition to bootstrap sampling of data points, a random subset of features ($\sqrt{p}$ or $\log_2(p)$) is selected at each candidate split. This decorrelates the trees and prevents dominant features from dictating every tree structure.

### Adaptive Boosting (AdaBoost)
A sequential boosting algorithm where weak learners (typically decision stumps) are trained iteratively. Incorrectly classified samples are assigned higher sample weights in subsequent iterations, forcing the ensemble to focus on difficult edge cases.

### Extreme Gradient Boosting (XGBoost)
A scalable tree boosting system optimizing custom objective functions via 2nd-order Taylor expansions ($g_i$ gradients and $h_i$ hessians). Features column block subsampling, exact/approximate split finding, built-in sparsity awareness, and explicit tree regularization ($\gamma$ leaf complexity and $\lambda$ $L_2$ leaf weights).

### Support Vector Machines (SVM) & Sequential Minimal Optimization (SMO)
A maximum-margin classifier that finds the optimal separating hyperplane maximizing the geometric margin between classes. Handled via the Dual Lagrangian formulation:
$$\max_\alpha \sum_{i=1}^n \alpha_i - \frac{1}{2}\sum_{i,j} \alpha_i \alpha_j y_i y_j K(x_i, x_j) \quad \text{s.t. } 0 \le \alpha_i \le C, \sum \alpha_i y_i = 0$$
SMO solves this large quadratic programming problem analytically by optimizing pairs of Lagrange multipliers ($\alpha_1, \alpha_2$) at each step without matrix inversion.

### Gaussian Naive Bayes
A probabilistic classifier applying Bayes' Theorem under the "naive" assumption of conditional independence among features given the class label. Feature likelihoods $P(x_j | y)$ are modeled using the Gaussian normal probability density function.

---

## 3. Data Engineering, Validation & Contracts

### Data Contracts
A formal agreement and specification between data producers and consumers guaranteeing that ingested data schemas, types, ranges, nullability constraints, and distributions adhere strictly to agreed standards before triggering downstream training or scoring.

### Pandera
A Python statistical data validation library designed for dataframes. Enforces declarative schema contracts, column data types, value ranges, and custom statistical check functions with detailed diagnostic error traces on validation failures.

### Feature Drift
The statistical deviation of input feature distributions $P(X)$ between training data and real-time production inference traffic over time, occurring even when the conditional probability $P(Y|X)$ remains unchanged.

### Concept Drift
The statistical change in the relationship between input features and target labels ($P(Y|X)$ changes over time), causing model accuracy and calibration to decay in production.

---

## 4. Experiment Tracking, Governance & Metadata

### MLflow Tracking
An open-source platform component for logging parameters, code versions, metrics, time-series curves, and binary model artifacts during ML training runs across distributed environments.

### MLflow Model Registry
A centralized model store and governance system providing a collaborative lifecycle management workflow with semantic versioning, stage transitions (`None` $\to$ `Staging` $\to$ `Production` $\to$ `Archived`), and deployment lineage.

### Model Governance & Lineage
The auditable tracking of every production artifact back to its exact training dataset version, code commit SHA, preprocessing pipeline, hyperparameter configuration, and test metrics.

---

## 5. Cloud Infrastructure, Networking & Containers

### Infrastructure as Code (IaC)
The practice of managing, provisioning, and version-controlling cloud infrastructure (servers, networks, firewalls, storage) through machine-readable definition files (e.g. HashiCorp Terraform / HCL) rather than manual console interaction.

### Virtual Private Cloud (VPC)
A logically isolated virtual network within a public cloud provider (such as AWS) where compute instances, databases, and load balancers are launched with dedicated IP CIDR blocks, subnets, and routing tables.

### Security Group (SG)
A stateful virtual firewall controlling inbound and outbound network traffic to compute instances (e.g. allowing HTTP port 8000, Prometheus port 9090, Grafana port 3000, and SSH port 22).

### OpenID Connect (OIDC) & AWS STS
A federated authentication protocol enabling external CI/CD pipelines (like GitHub Actions) to authenticate securely with AWS using short-lived Security Token Service (STS) credentials without storing long-lived AWS Access Keys in repository secrets.

### Multi-Stage Docker Builds
A Docker build pattern separating the compilation/dependency-installation environment (builder stage with compilers, gcc, build-essential) from the minimal execution environment (runtime stage with lean python-slim and non-root users), reducing image footprint and attack surface.

### Serverless Inference (AWS Lambda)
An event-driven execution model where containerized ML inference handlers execute on-demand in response to API requests without managing underlying virtual machines, scaling automatically from 0 to thousands of concurrent requests.

---

## 6. Automated CI/CD/CT Pipelines & Quality Gates

### Continuous Integration (CI)
The automated workflow of building, linting, and unit-testing code changes on every pull request or commit to detect regression errors early in the development lifecycle.

### Continuous Deployment (CD)
The automated release pipeline that packages approved, tested application containers and deploys them to target staging/production environments (EC2, ECS, Lambda) without manual intervention.

### Continuous Training (CT)
An MLOps-specific paradigm where training and evaluation pipelines automatically retrain, validate, and register new model versions when incoming production data drift or scheduled triggers occur.

### Quality Gate
A hard threshold or check (e.g., Ruff linting pass, 100% Pytest pass rate, minimum model ROC-AUC $\ge 0.75$, maximum inference latency $\le 50\text{ms}$) that must be satisfied before a pipeline allows code or artifacts to advance to downstream deployment.

---

## 7. Observability, Telemetry & Load Testing

### Prometheus & PromQL
A time-series database and monitoring ecosystem that scrapes numeric metrics via HTTP GET requests (`/metrics`) using pull-based telemetry. PromQL is its functional query language used to compute rates, latency histograms, and alert thresholds.

### Percentile Latencies (p50, p95, p99)
Statistical measures of latency distribution:
- **p50 (Median)**: 50% of all requests are answered faster than this value.
- **p95**: 95% of requests complete within this duration; highlights user experience for the majority of traffic.
- **p99**: The 99th percentile capturing long-tail tail latency and worst-case queuing delays.

### Head-of-Line Blocking (FastAPI Async vs Sync)
A bottleneck occurring when CPU-bound synchronous code (such as Pandas transformations and XGBoost matrix predictions) is executed inside an `async def` route, freezing the single asyncio event loop thread and causing concurrent HTTP requests to queue up. Resolved by declaring routes as standard `def` to trigger automated worker threadpool dispatching.

### Locust
An open-source Python-based distributed load testing framework that allows defining user personas with custom task weights, wait times, dynamic payload generators, and event hooks to stress-test APIs and benchmark real-time system stability.

### Service Level Agreement (SLA)
A formal technical contract defining acceptable performance and reliability bounds (e.g. Error Rate $< 0.5\%$, Throughput $\ge 30\text{ RPS}$, and $\text{p95 Latency} \le 120\text{ms}$).
