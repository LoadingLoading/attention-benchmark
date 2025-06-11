from fvcore.nn import FlopCountAnalysis
import torch

def measure_flops(model, input_tensor):
    tensor_clone = input_tensor.clone()
    #print(f"input tensor for FLOP measurement: {tensor_clone.shape}")
    if len(tensor_clone.shape) > 3:
        #print("should be reshape before measure flops")
        tensor_clone = tensor_clone.view(tensor_clone.size(0), -1, tensor_clone.size(-1))
        #print(f"Reshaped input tensor for FLOP measurement: {tensor_clone.shape}")
    
    flops = FlopCountAnalysis(model, tensor_clone)
    return flops.total()



