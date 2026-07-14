import json
from pathlib import Path
import torch
import torch.nn as nn

class ONNXWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, images, telemetry, physics):
        out = self.model(images, telemetry, physics)
        return out['predicted_image'], out['class_logits'], out['reg_output']


class ModelExporter:
    def __init__(self, model: nn.Module, output_dir: str = 'models/vision'):
        self.model = model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def export_pytorch(self, checkpoint_data: dict, filename: str = 'best.pt') -> str:
        out_file = self.output_dir / filename
        torch.save(checkpoint_data, out_file)
        return str(out_file)
    
    def export_onnx(
        self,
        filename: str = 'best.onnx',
        seq_len: int = 4,
        img_size: int = 512,
        opset_version: int = 17,
    ) -> str:
        out_file = self.output_dir / filename
        
        self.model.eval()
        wrapper = ONNXWrapper(self.model)
        
        dummy_images = torch.randn(1, seq_len, 3, img_size, img_size)
        dummy_telemetry = torch.randn(1, 10)
        dummy_physics = torch.randn(1, 5)
        
        torch.onnx.export(
            wrapper,
            (dummy_images, dummy_telemetry, dummy_physics),
            out_file,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['images', 'telemetry', 'physics'],
            output_names=['predicted_image', 'class_logits', 'reg_output'],
            dynamic_axes={
                'images': {0: 'batch_size'},
                'telemetry': {0: 'batch_size'},
                'physics': {0: 'batch_size'},
                'predicted_image': {0: 'batch_size'},
                'class_logits': {0: 'batch_size'},
                'reg_output': {0: 'batch_size'}
            }
        )
        return str(out_file)
    
    def export_torchscript(
        self,
        filename: str = 'best.torchscript',
        seq_len: int = 4,
        img_size: int = 512,
    ) -> str:
        out_file = self.output_dir / filename
        
        self.model.eval()
        dummy_images = torch.randn(1, seq_len, 3, img_size, img_size)
        dummy_telemetry = torch.randn(1, 10)
        dummy_physics = torch.randn(1, 5)
        
        traced_script_module = torch.jit.trace(self.model, (dummy_images, dummy_telemetry, dummy_physics), strict=False)
        traced_script_module.save(out_file)
        
        return str(out_file)
    
    def save_model_config(self, config: dict, filename: str = 'model_config.json') -> str:
        out_file = self.output_dir / filename
        with open(out_file, 'w') as f:
            json.dump(config, f, indent=4)
        return str(out_file)
    
    def save_training_metadata(self, metadata: dict, filename: str = 'training_metadata.json') -> str:
        out_file = self.output_dir / filename
        with open(out_file, 'w') as f:
            json.dump(metadata, f, indent=4)
        return str(out_file)
    
    def save_feature_schema(self, filename: str = 'feature_schema.json') -> str:
        schema = {
            "inputs": {
                "images": {"shape": ["batch", "seq_len", 3, 512, 512], "dtype": "float32"},
                "telemetry": {"shape": ["batch", 10], "dtype": "float32"},
                "physics": {"shape": ["batch", 5], "dtype": "float32"}
            },
            "outputs": {
                "predicted_image": {"shape": ["batch", 3, 512, 512], "dtype": "float32"},
                "class_logits": {"shape": ["batch", 5], "dtype": "float32"},
                "reg_output": {"shape": ["batch", 1], "dtype": "float32"}
            }
        }
        out_file = self.output_dir / filename
        with open(out_file, 'w') as f:
            json.dump(schema, f, indent=4)
        return str(out_file)
    
    def export_all(
        self,
        checkpoint_data: dict,
        config: dict,
        metadata: dict,
        seq_len: int = 4,
        img_size: int = 512,
    ) -> dict:
        return {
            "pytorch": self.export_pytorch(checkpoint_data),
            "onnx": self.export_onnx(seq_len=seq_len, img_size=img_size),
            "torchscript": self.export_torchscript(seq_len=seq_len, img_size=img_size),
            "config": self.save_model_config(config),
            "metadata": self.save_training_metadata(metadata),
            "schema": self.save_feature_schema()
        }
