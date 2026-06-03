# Provider Details

## Amazon Web Services (AWS)

### GPU Instance Families
- **P5** — NVIDIA H100 (8 GPU, 192 vCPUs, 2TB RAM)
- **P4** — NVIDIA A100 (8 GPU, 96 vCPUs, 1.15TB RAM)
- **G6** — NVIDIA L4 (1-4 GPU, inference optimized)
- **G5** — NVIDIA A10G (1-8 GPU, versatile)
- **G6** — NVIDIA T4 (1 GPU, budget inference)

### Key Features
- **EFA (Elastic Fabric Adapter)** for multi-node training
- **FSx for Lustre** high-performance storage
- **SageMaker** managed ML platform
- **Spot Instances** — up to 70% savings
- **Reserved Instances** — 1 or 3 year commitments

### Spot Availability
H100 spot: ~$49/hr (50% off on-demand)
A100 spot: ~$16/hr (50% off)

### Regions with GPU
us-east-1, us-west-2, eu-west-1, ap-southeast-1, ap-northeast-1

---

## Google Cloud Platform (GCP)

### GPU Instance Families
- **A3** — NVIDIA H100 (8 GPU, MegaGPU available)
- **A2** — NVIDIA A100 (1-8 GPU, multiple configs)
- **G2** — NVIDIA L4 (inference optimized)
- **N1** — NVIDIA T4 (budget, preemptible available)

### Key Features
- **TPU v4/v5e** alternatives to GPUs
- **Preemptible VMs** — up to 70% savings (best discounts)
- **Vertex AI** managed ML platform
- **GCS** integration for data pipelines
- **Multi-Region** deployment options

### Spot/Preemptible Availability
Best spot discounts in the industry — up to 70% off on-demand
H100 preemptible: ~$28/hr (70% off)
A100 preemptible: ~$9/hr (70% off)

### Regions with GPU
us-central1, us-west1, europe-west1, asia-east1, us-east4

---

## Microsoft Azure

### GPU Instance Families
- **ND H100 v5** — NVIDIA H100 (8 GPU, InfiniBand)
- **ND A100 v4** — NVIDIA A100 (8 GPU)
- **NC A100 v4** — NVIDIA A100 (1 GPU, cost-effective)
- **NC T4 v3** — NVIDIA T4 (budget)
- **NV A10 v5** — NVIDIA A10 (inference)

### Key Features
- **InfiniBand** interconnect for multi-node
- **Azure ML** managed platform
- **Spot VMs** — up to 60% savings
- **Hybrid Cloud** with Azure Arc
- **OpenAI** integration for API workloads

### Spot Availability
H100 spot: ~$40/hr (60% off)
A100 spot: ~$11/hr (60% off)

### Regions with GPU
East US, West US 2, West Europe, Southeast Asia, Japan East

---

## DigitalOcean

### GPU Droplet Types
- **H100** (1x or 8x) — $5.39/hr or $43.10/hr
- **A100** (1x or 8x) — $2.89/hr or $23.10/hr
- **A10** (1x) — $1.10/hr
- **RTX6000** (1x or 4x) — $0.76/hr or $3.04/hr

### Key Features
- **Simple pricing** — no spot market complexity
- **No egress fees** — data transfer included
- **Droplet API** — simple provisioning
- **Kubernetes** managed GPU nodes
- **Developer-focused** UX

### Best For
- Small teams wanting simplicity
- No spot market complexity
- Predictable costs without egress surprises

---

## Akamai Cloud Computing (Linode)

### GPU Types
- **H100** (1x or 8x) — $4.99/hr or $39.90/hr
- **A100** (1x or 8x) — $2.49/hr or $19.90/hr
- **RTX6000** (1x) — $0.65/hr
- **L4** (1x) — $1.10/hr
- **T4** (1x) — $0.50/hr

### Key Features
- **Competitive pricing** — among lowest rates
- **Simple plans** — no hidden costs
- **Akamai CDN** integration
- **Growing GPU fleet** — expanding availability
- **Predictable billing**

### Best For
- Budget-conscious teams
- Inference workloads
- Edge + cloud combinations with Akamai CDN
