# Cost Optimization Strategies

## 1. Spot/Preemptible Instances

**Potential savings: 40-70%**

Best for:
- Fault-tolerant training with checkpointing
- Batch inference jobs
- Hyperparameter sweeps
- Any workload with pause/resume capability

| Provider | Spot Discount | Interruption Rate |
|----------|--------------|-------------------|
| AWS | 50-70% | Medium |
| GCP (Preemptible) | 60-70% | High |
| Azure | 50-70% | Medium |
| DigitalOcean | N/A | N/A |
| Akamai | N/A | N/A |

### Implementation Tips
1. Implement checkpointing every 5-15 minutes
2. Use multiple availability zones
3. Set maximum spot price to protect against extreme costs
4. Mix spot + on-demand for resilience

---

## 2. Right-Sizing Instances

**Potential savings: 20-50%**

Common mistakes:
- Using H100 when T4/L4 suffices for inference
- Provisioning 8 GPUs when 4 would do
- Over-provisioning CPU/RAM alongside GPUs

### GPU Selection Guide

**Inference workloads:**
- <10B params → T4/L4 ($1-2/hr)
- 10-30B params → A10/A10G ($3-4/hr)
- 30-70B params → A100 ($3-15/hr)
- 70B+ params → H100 ($5-100/hr)

**Training workloads:**
- Fine-tuning <13B → A100 (1-2 GPU)
- Fine-tuning 13-70B → A100/H100 (4-8 GPU)
- Pre-training → H100 cluster (8+ GPU)

---

## 3. Reserved Instances

**Potential savings: 30-60% vs on-demand**

Best for:
- Always-on inference servers (>70% utilization)
- Long-running training pipelines
- Production deployments with predictable usage

### When to Use Reserved
- Compute utilization > 70%
- Commitment period > 6 months
- Need guaranteed capacity

---

## 4. Provider Optimization

### Cost by Use Case

**Cheapest for single GPU inference:**
- Akamai T4: $0.50/hr
- DigitalOcean RTX6000: $0.76/hr

**Cheapest for 8x H100 training:**
- DigitalOcean: $43.10/hr
- Akamai: $39.90/hr
- GCP spot: $28.24/hr

**Cheapest for 8x A100 training:**
- Akamai: $19.90/hr
- DigitalOcean: $23.10/hr
- GCP spot: $8.82/hr

### Regional Savings
- US regions typically cheapest
- Asia/Pacific often 10-20% more expensive
- Europe 5-15% more than US

---

## 5. Mixed Precision & Optimization

**Potential savings: 50% on compute time**

```python
# Use mixed precision for training
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    output = model(input)
    loss = loss_fn(output, target)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

Benefits:
- BF16/FP16 training uses half the memory
- 1.5-2x throughput improvement
- Minimal accuracy loss

---

## 6. Batching & Throughput Optimization

**Potential savings: 2-5x throughput improvement**

- Use dynamic batching for inference
- Maximize GPU utilization to >80%
- Use vLLM or TensorRT for serving
- Implement request queuing

---

## 7. Storage & Transfer Optimization

**Potential savings: 30-80% on data costs**

- Use S3/GCS lifecycle policies
- Compress training data
- Use regional storage when possible
- Minimize cross-region transfers
- Consider checkpoint storage strategies

---

## Summary Checklist

- [ ] Benchmark your workload on different GPU types
- [ ] Check spot pricing for your target region
- [ ] Evaluate reserved pricing for long-running workloads
- [ ] Implement mixed precision training
- [ ] Right-size your instances
- [ ] Use batching for inference
- [ ] Monitor GPU utilization
- [ ] Implement checkpointing for spot resilience
- [ ] Compare at least 3 providers
- [ ] Factor in all costs (storage, transfer, support)
