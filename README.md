# Sentinel-Vision

**Autonomous Real-Time Defect Detection & Compliance System**

## Overview
Sentinel-Vision is a high-speed defect detection system designed for manufacturing lines. It utilizes advanced CNN architectures (EfficientNet-V2 / Vision Transformer) to identify structural defects (e.g., micro-cracks, solder bridges) and provides explainable AI reports using Grad-CAM.

## Key Features
- **Advanced AI Engine**: Powered by EfficientNet-V2 / ViT with Mixed Precision Training.
- **Fine-Grained Recognition**: Classification of specific defect types.
- **Explainability**: Grad-CAM heatmaps to visualize model focus.
- **Professional Data Engineering**: DVC for version control, Stable Diffusion for synthetic data.
- **MLOps & Deployment**: FastAPI, Docker, Kubernetes (HPA), and Prometheus monitoring.

## Setup
1. Create a virtual environment: `python -m venv venv`
2. Activate: `.\venv\Scripts\activate`
3. Install dependencies: `pip install -r requirements.txt`

## System Architecture

```mermaid
graph TD
    subgraph Edge_Device
        A[High-Speed Camera] -->|Images| B(API Gateway / FastAPI)
    end
    
    subgraph Kubernetes_Cluster
        B -->|Load Balanced| C{K8s Service}
        C --> D[Model Pod 1]
        C --> E[Model Pod 2]
        C --> F[Model Pod N]
        
        D & E & F -->|Inference| G[EfficientNet-V2 / ViT]
        G -->|Grad-CAM| H[Explainability Module]
    end
    
    subgraph Data_Pipeline
        I[Data Lake (S3/GCS)] -->|DVC Pull| J[Training Node]
        J -->|Train & Eval| K[Model Registry]
        K -->|Deploy| D & E & F
    end
    
    H -->|JSON Result| L[Dashboard / Compliance DB]
```

## Performance Metrics (Target)

| Metric | Target Value | Notes |
| :--- | :--- | :--- |
| **F1-Score** | > 0.95 | Critical for minimizing false positives/negatives in manufacturing. |
| **Inference Latency** | < 50ms | Required for real-time processing on high-speed lines. |
| **GPU Utilization** | > 90% | Achieved via Mixed Precision Training (FP16) and optimal batching. |
| **Throughput** | 100+ FPS | Scalable via Horizontal Pod Autoscaler (HPA). |

## Cost Analysis (Cloud Estimate)

**Scenario**: 24/7 Deployment on AWS with auto-scaling.

*   **Inference Nodes (G4dn.xlarge - T4 GPU)**:
    *   Cost: ~$0.526/hour (On-Demand)
    *   Monthly (1 node): ~$380
    *   *Optimization*: Use Spot Instances for handled spikes (~70% savings).
*   **Training (P3.2xlarge - V100 GPU)**:
    *   Cost: ~$3.06/hour
    *   Frequency: Retrain weekly (approx. 5 hours) -> ~$60/month.
*   **Storage (S3)**:
    *   Dataset (1TB): ~$23/month.
*   **Total Estimated**: ~$463/month (Baseline)

*This architecture allows for significant cost reduction by scaling down inference nodes during non-production hours.*

