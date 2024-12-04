"""
model_logger.py - Utility for logging PyTorch model summaries
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
from collections import OrderedDict
import numpy as np
import sys
from io import StringIO
from contextlib import contextmanager
import logging
from typing import Tuple, Optional, Union

@contextmanager
def capture_stdout():
    """Capture stdout and return it as a string."""
    stdout = StringIO()
    old_stdout = sys.stdout
    sys.stdout = stdout
    try:
        yield stdout
    finally:
        sys.stdout = old_stdout

def _register_hook(module, hooks, summary, batch_size):
    """Register forward hook for the module."""
    def hook(module, input, output):
        class_name = str(module.__class__).split(".")[-1].split("'")[0]
        module_idx = len(summary)

        m_key = "%s-%i" % (class_name, module_idx + 1)
        summary[m_key] = OrderedDict()
        summary[m_key]["input_shape"] = list(input[0].size())
        summary[m_key]["input_shape"][0] = batch_size
        
        if isinstance(output, (list, tuple)):
            summary[m_key]["output_shape"] = [
                [-1] + list(o.size())[1:] for o in output
            ]
        else:
            summary[m_key]["output_shape"] = list(output.size())
            summary[m_key]["output_shape"][0] = batch_size

        params = 0
        if hasattr(module, "weight") and hasattr(module.weight, "size"):
            params += torch.prod(torch.LongTensor(list(module.weight.size())))
            summary[m_key]["trainable"] = module.weight.requires_grad
        if hasattr(module, "bias") and hasattr(module.bias, "size"):
            params += torch.prod(torch.LongTensor(list(module.bias.size())))
        summary[m_key]["nb_params"] = params

    if (
        not isinstance(module, nn.Sequential)
        and not isinstance(module, nn.ModuleList)
        and not (module == model)
    ):
        hooks.append(module.register_forward_hook(hook))

def log_model_summary(
    model: nn.Module,
    input_size: Union[Tuple, list],
    logger: Optional[logging.Logger] = None,
    batch_size: int = -1,
    device: str = "cuda"
) -> None:
    """
    Log a summary of the PyTorch model architecture and parameters.
    
    Args:
        model: PyTorch model to summarize
        input_size: Size of input tensor (excluding batch size)
        logger: Logger instance to use. If None, creates a default logger
        batch_size: Batch size to use for summary. Defaults to -1
        device: Device to run the model on ('cuda' or 'cpu')
    
    Example:
        >>> logger = logging.getLogger(__name__)
        >>> input_size = (3, 224, 224)  # For an image model
        >>> log_model_summary(model, input_size, logger)
    """
    if logger is None:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            ))
            logger.addHandler(handler)

    device = device.lower()
    assert device in ["cuda", "cpu"], "Device must be 'cuda' or 'cpu'"

    if device == "cuda" and torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    # Handle multiple inputs if necessary
    if isinstance(input_size, tuple):
        input_size = [input_size]

    # Create input tensor
    x = [torch.rand(2, *in_size).type(dtype) for in_size in input_size]

    # Initialize summary dict and hooks list
    summary = OrderedDict()
    hooks = []

    # Register hooks
    model.apply(lambda module: _register_hook(module, hooks, summary, batch_size))

    # Make a forward pass
    with capture_stdout():  # Suppress the forward pass output
        model(*x)

    # Remove hooks
    for h in hooks:
        h.remove()

    # Log the summary
    logger.info("-" * 64)
    logger.info("{:>20}  {:>25} {:>15}".format(
        "Layer (type)", "Output Shape", "Param #"
    ))
    logger.info("=" * 64)

    total_params = 0
    total_output = 0
    trainable_params = 0

    for layer in summary:
        line = "{:>20}  {:>25} {:>15}".format(
            layer,
            str(summary[layer]["output_shape"]),
            "{0:,}".format(summary[layer]["nb_params"])
        )
        logger.info(line)
        
        total_params += summary[layer]["nb_params"]
        total_output += np.prod(summary[layer]["output_shape"])
        if "trainable" in summary[layer]:
            if summary[layer]["trainable"]:
                trainable_params += summary[layer]["nb_params"]

    # Calculate sizes
    total_input_size = abs(np.prod(input_size) * batch_size * 4. / (1024 ** 2.))
    total_output_size = abs(2. * total_output * 4. / (1024 ** 2.))
    total_params_size = abs(total_params.numpy() * 4. / (1024 ** 2.))
    total_size = total_params_size + total_output_size + total_input_size

    logger.info("=" * 64)
    logger.info("Total params: {:,}".format(total_params))
    logger.info("Trainable params: {:,}".format(trainable_params))
    logger.info("Non-trainable params: {:,}".format(total_params - trainable_params))
    logger.info("-" * 64)
    logger.info("Input size (MB): {:.2f}".format(total_input_size))
    logger.info("Forward/backward pass size (MB): {:.2f}".format(total_output_size))
    logger.info("Params size (MB): {:.2f}".format(total_params_size))
    logger.info("Estimated Total Size (MB): {:.2f}".format(total_size))
    logger.info("-" * 64)