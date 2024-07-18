from common.registry import registry
from models.base_model import BaseModel
from models.lion_t5 import LIONT5InstructAdapter

__all__ = [
    "LIONT5InstructAdapter"
]

def load_model(name, model_type, is_eval=False, device="cpu"):
    """
    Load supported models.

    Args:
        name (str): name of the model.
        model_type (str): type of the model.
        is_eval (bool): whether the model is in eval mode. Default: False.
        device (str): device to use. Default: "cpu".

    Returns:
        model (torch.nn.Module): model.
    """

    model = registry.get_model_class(name).from_pretrained(model_type=model_type)

    if is_eval:
        model.eval()

    if device == "cpu":
        model = model.float()

    return model.to(device)