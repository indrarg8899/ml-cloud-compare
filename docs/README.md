# ML Cloud Compare Documentation

## Providers Supported

### Major Cloud
- **AWS** (EC2 P4d, P5, G5, Trn1)
- **Azure** (ND H100 v5, NC A100 v4)
- **GCP** (A3, A2, G2)

### GPU Cloud
- **Lambda** (GPU Cloud)
- **CoreWeave** (GPU Cloud)
- **vast.ai** (Marketplace)

## Workload Profiles

### LLM Training (7B)
- Batch size: 32
- Sequence length: 2048
- dtype: FP16
- Estimated memory: ~50 GB

### LLM Fine-tuning (7B)
- Batch size: 8
- Sequence length: 4096
- dtype: FP16 (LoRA)
- Estimated memory: ~20 GB

### Vision Training
- Batch size: 128
- Input: 224x224
- dtype: FP32
- Estimated memory: ~8 GB

## Cost Factors

1. **GPU hourly rate** - Base compute cost
2. **Storage** - Model and checkpoint storage
3. **Network** - Data transfer fees
4. **Support** - Enterprise support tiers
5. **Reserved pricing** - 1yr/3yr commitments (30-50% savings)

## Tips

- Use spot/preemptible instances for training (60-70% savings)
- AMD MI300X often cheaper per TFLOP than NVIDIA H100
- Check provider-specific ML frameworks (SageMaker, Vertex AI)
- Consider total training time, not just hourly rate
