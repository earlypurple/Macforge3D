from typing import Any, List, Literal, Optional, Union, Dict, Tuple
import torch

class ShapEPipeline:
    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        torch_dtype: Optional[torch.dtype] = None,
        **kwargs: Any
    ) -> "ShapEPipeline":
        ...
    
    def to(self, device: Union[str, torch.device]) -> "ShapEPipeline":
        ...
    
    def __call__(
        self,
        prompt: str,
        guidance_scale: float = 15.0,
        num_inference_steps: int = 64,
        frame_size: int = 256,
        output_type: Literal["mesh", "image"] = "mesh",
        **kwargs: Any
    ) -> Dict[str, List[Any]]:
        ...
