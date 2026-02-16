# Testing Commands Reference

## Quick Commands for C-CLIP Testing

### 1. Quick Test (Fastest)
```powershell
.venv\Scripts\python.exe test_model.py
```
- Tests both task_1 and task_2 validation sets
- Shows all retrieval metrics (Recall@1/5/10)
- Takes ~30 seconds

### 2. Full Evaluation with JSON Export
```powershell
.venv\Scripts\python.exe src\evaluate.py `
  --checkpoint checkpoints\sample_test\model_after_task_0.pt `
  --config configs\sample_config.yaml `
  --eval_config configs\sample_eval_config.json `
  --output results\sample_evaluation.json
```
- Saves results to JSON file
- Takes ~40 seconds

### 3. View Results
```powershell
# View JSON results
Get-Content results\sample_evaluation.json | ConvertFrom-Json | ConvertTo-Json -Depth 10

# View training report
Get-Content TRAINING_REPORT.md
```

### 4. Verify Checkpoint
```powershell
# Check if checkpoint exists
Test-Path checkpoints\sample_test\model_after_task_0.pt

# View checkpoint details
Get-Item checkpoints\sample_test\model_after_task_0.pt | Format-List
```

### 5. Dataset Verification
```powershell
# Count validation samples
(Get-Content data\task_1\val.csv).Count - 1
(Get-Content data\task_2\val.csv).Count - 1

# View sample data
Get-Content data\task_1\val.csv | Select-Object -First 5
Get-Content data\task_2\val.csv | Select-Object -First 5
```

### 6. GPU Monitoring (While Testing)
```powershell
# In a separate terminal
nvidia-smi -l 1
```

### 7. Performance Benchmark
```powershell
# Time the test execution
Measure-Command { .venv\Scripts\python.exe test_model.py }
```

### 8. Interactive Testing (Python REPL)
```powershell
.venv\Scripts\python.exe
```

Then in Python:
```python
import torch
from src.models.cclip import CCLIP

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = CCLIP(
    clip_model_name='ViT-B-32',
    pretrained='openai',
    lora_r=8,
    lora_alpha=16,
    device=device
).to(device)

model.load_checkpoint('checkpoints/sample_test/model_after_task_0.pt')
print(f"Model loaded! Current task: {model.current_task}")
```

## Expected Results

### Task 1 (after training both tasks):
- **I2T Recall@1:** 10.0%
- **I2T Recall@5:** 60.0%
- **I2T Recall@10:** 100.0%
- **T2I Recall@1:** 0.0%
- **T2I Recall@5:** 60.0%
- **T2I Recall@10:** 100.0%

### Task 2:
- **I2T Recall@1:** 0.0%
- **I2T Recall@5:** 50.0%
- **I2T Recall@10:** 100.0%
- **T2I Recall@1:** 20.0%
- **T2I Recall@5:** 50.0%
- **T2I Recall@10:** 100.0%

## Output Files

- `results/sample_evaluation.json` - Detailed metrics in JSON format
- `TRAINING_REPORT.md` - Training statistics and analysis
- `checkpoints/sample_test/model_after_task_0.pt` - Trained model checkpoint (580 MB)

## Troubleshooting

### If checkpoint not found:
```powershell
# Re-run training
.venv\Scripts\python.exe src\train.py --config configs\sample_config.yaml
```

### If CUDA out of memory:
```powershell
# Edit configs\sample_config.yaml and reduce batch_size to 8 or 4
```

### Check Python environment:
```powershell
.venv\Scripts\python.exe --version
.venv\Scripts\pip list | Select-String "torch|open-clip"
```

## Summary Statistics

- **Model Size:** 580 MB checkpoint
- **Total Parameters:** 152M (Task 1), 303M (Task 2 with old_clip)
- **Trainable Parameters:** 771K (0.5%)
- **Architecture:** ViT-B/32 + LoRA (rank=8)
- **Training Time:** ~2 minutes per task (5 epochs)
- **Testing Time:** ~30 seconds for both tasks
- **Hardware:** NVIDIA RTX 3050 6GB Laptop GPU
