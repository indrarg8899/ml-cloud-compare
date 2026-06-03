# Methodology

## How We Collect Pricing Data

### Sources
All pricing data comes from official cloud provider pricing pages:
- **AWS**: [EC2 Pricing](https://aws.amazon.com/ec2/pricing/) — us-east-1 region
- **GCP**: [Compute Engine Pricing](https://cloud.google.com/compute/gpus-pricing) — us-central1
- **Azure**: [VM Pricing](https://azure.microsoft.com/en-us/pricing/details/virtual-machines/linux/) — East US
- **DigitalOcean**: [GPU Droplet Pricing](https://www.digitalocean.com/pricing/gpu-droplets) — nyc1
- **Akamai**: [Linode Pricing](https://www.linode.com/pricing/) — us-east

### Update Frequency
- Prices verified weekly against official pages
- Spot prices sampled daily (they fluctuate)
- Reserved pricing updated monthly
- CSV datasets updated with each release

### Pricing Methodology

#### On-Demand Pricing
- Uses standard pay-as-you-go hourly rates
- All prices in USD
- Excludes taxes and enterprise discounts
- Represents list price without negotiation

#### Spot/Preemptible Pricing
- Spot prices are estimates based on recent market rates
- Actual spot prices fluctuate based on supply/demand
- Spot prices shown are ~60-70% of on-demand (typical range)
- Spot interruptions not factored into cost estimates
- User must build in checkpointing and restart overhead

#### Reserved Pricing
- Based on 1-year commitment with no upfront
- Savings compared to on-demand pricing
- Does not include partial or all-upfront payment options
- Reserved pricing not available on all instance types

### Performance Methodology

#### Benchmark Data
Performance data derived from:
1. **MLPerf Inference v4.0** results
2. **MLPerf Training v3.1** results
3. Vendor-published peak TFLOPS specifications
4. Independent benchmark sources (Lambda Labs, AnandTech)
5. Synthetic benchmarks where official data unavailable

#### Performance-per-Dollar Calculation
```
composite_score = (
    tflops_per_dollar_normalized * 0.4 +
    inference_per_dollar_normalized * 0.4 +
    memory_per_dollar_normalized * 0.2
)
```

Normalization uses H100 as baseline (score = 1.0).

### Cost Estimation Model

#### Training Cost
```
total_cost = (hourly_rate × hours × gpu_count) + storage_cost + transfer_cost
storage_cost = storage_gb × $/GB/month × (hours / 720)
transfer_cost = data_gb × $/GB
```

#### Inference Cost
```
monthly_cost = hourly_rate × hours_per_day × days_per_month × gpu_count
```

### Caveats
- Prices change frequently; verify with providers
- Regional pricing varies significantly
- Enterprise discounts not reflected
- Spot prices are estimates
- Performance may vary with workload characteristics
- Network and storage costs not always included
