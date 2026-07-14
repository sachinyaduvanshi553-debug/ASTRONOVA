import json
from pathlib import Path

class ReportGenerator:
    def __init__(self, output_dir: str = 'reports/vision'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_training_report(
        self, 
        training_history: dict, 
        model_config: dict,
        dataset_stats: dict,
        output_path: str = 'training_report.md'
    ) -> str:
        out_file = self.output_dir / output_path
        
        best_epoch = training_history.get('best_epoch', 0)
        best_loss = training_history.get('best_loss', 0.0)
        
        md = f"""# 📈 AstroNova Vision Training Report

## Configuration
- **Model Architecture**: {model_config.get('architecture', 'SolarVisionPredictor')}
- **Epochs Trained**: {training_history.get('total_epochs', 0)}
- **Best Epoch**: {best_epoch}
- **Best Val Loss**: {best_loss:.4f}

## Dataset
- **Total Samples**: {dataset_stats.get('total_samples', 0)}
- **Sequence Length**: {dataset_stats.get('seq_len', 4)}
- **Prediction Horizon**: {dataset_stats.get('horizon', 60)} minutes

## Final Metrics
```json
{json.dumps(training_history.get('final_metrics', {}), indent=2)}
```
"""
        with open(out_file, 'w') as f:
            f.write(md)
        return str(out_file)
    
    def generate_evaluation_report(
        self,
        eval_results: dict,
        output_path: str = 'evaluation_report.md'
    ) -> str:
        out_file = self.output_dir / output_path
        
        summary = eval_results.get('summary', {})
        
        md = f"""# 📊 AstroNova Vision Evaluation Report

## Image Quality Metrics
| Metric | Mean | Std Dev | Min | Max |
|--------|------|---------|-----|-----|
| PSNR | {summary.get('psnr', {}).get('mean', 0):.2f} dB | {summary.get('psnr', {}).get('std', 0):.2f} | {summary.get('psnr', {}).get('min', 0):.2f} | {summary.get('psnr', {}).get('max', 0):.2f} |
| SSIM | {summary.get('ssim', {}).get('mean', 0):.4f} | {summary.get('ssim', {}).get('std', 0):.4f} | {summary.get('ssim', {}).get('min', 0):.4f} | {summary.get('ssim', {}).get('max', 0):.4f} |
| MAE | {summary.get('mae', {}).get('mean', 0):.4f} | {summary.get('mae', {}).get('std', 0):.4f} | {summary.get('mae', {}).get('min', 0):.4f} | {summary.get('mae', {}).get('max', 0):.4f} |

## Classification Metrics
- Accuracy: {eval_results.get('accuracy', 0.0):.2%}
- Macro F1: {eval_results.get('macro_f1', 0.0):.4f}

"""
        with open(out_file, 'w') as f:
            f.write(md)
        return str(out_file)
    
    def generate_xai_report(
        self,
        xai_results: dict,
        output_path: str = 'xai_report.md'
    ) -> str:
        out_file = self.output_dir / output_path
        
        features = xai_results.get('feature_importances', {})
        
        md = f"""# 🧠 AstroNova Vision Explainability Report

## Telemetry & Physics Feature Importance
| Feature | Importance Score |
|---------|------------------|
"""
        for k, v in sorted(features.items(), key=lambda item: item[1], reverse=True):
            md += f"| {k} | {v:.4f} |\n"
            
        with open(out_file, 'w') as f:
            f.write(md)
        return str(out_file)
    
    def generate_uncertainty_report(
        self,
        uncertainty_stats: dict,
        output_path: str = 'uncertainty_report.md'
    ) -> str:
        out_file = self.output_dir / output_path
        
        md = f"""# 🎲 AstroNova Vision Uncertainty Report

## Monte Carlo Dropout Statistics
- Mean Confidence Score: {uncertainty_stats.get('mean_confidence', 0.0):.4f}
- Mean Class Entropy: {uncertainty_stats.get('mean_entropy', 0.0):.4f}
- Mean Flux StdDev: {uncertainty_stats.get('mean_flux_std', 0.0):.4e} W/m²
"""
        with open(out_file, 'w') as f:
            f.write(md)
        return str(out_file)
    
    def generate_benchmark_report(
        self,
        benchmark_results: dict,
        output_path: str = 'benchmark_report.md'
    ) -> str:
        out_file = self.output_dir / output_path
        
        md = f"""# 🏆 AstroNova Vision Benchmark Report

| Model | TSS | HSS | PSNR | SSIM |
|-------|-----|-----|------|------|
"""
        for model, res in benchmark_results.items():
            md += f"| {model} | {res.get('tss', 0):.3f} | {res.get('hss', 0):.3f} | {res.get('psnr', 0):.2f} | {res.get('ssim', 0):.4f} |\n"
            
        with open(out_file, 'w') as f:
            f.write(md)
        return str(out_file)
    
    def generate_all_reports(self, **kwargs) -> dict:
        reports = {}
        if 'training_history' in kwargs:
            reports['training'] = self.generate_training_report(
                kwargs['training_history'], kwargs.get('model_config', {}), kwargs.get('dataset_stats', {})
            )
        if 'eval_results' in kwargs:
            reports['evaluation'] = self.generate_evaluation_report(kwargs['eval_results'])
        if 'xai_results' in kwargs:
            reports['xai'] = self.generate_xai_report(kwargs['xai_results'])
        if 'uncertainty_stats' in kwargs:
            reports['uncertainty'] = self.generate_uncertainty_report(kwargs['uncertainty_stats'])
        if 'benchmark_results' in kwargs:
            reports['benchmark'] = self.generate_benchmark_report(kwargs['benchmark_results'])
            
        return reports
