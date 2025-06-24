import sys
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import pytest
import torch
from torch.testing import assert_close
torch.use_deterministic_algorithms(True)

# Import reference and student's implementation
from test_utils import ResultCapture, score_results

# get student repo path from env
student_repo_path = os.getenv("STUDENT_REPO_PATH", None)
if student_repo_path is None:
    print("env variable `STUDENT_REPO_PATH` is not set")
    sys.exit(1)
sys.path.insert(0, os.path.abspath(os.path.join(student_repo_path)))

# import reference module
from ref import matmul_with_importance as matmul_with_importance_ref

# import student's source module
from src import matmul_with_importance


# constants for all score test cases
ATOL = 1e-5
RTOL = 1e-5
SEED = 142
TIMEOUT = 20

# define test modes
test_modes = [
    "fwd", # to test if the implementation is correct when only forward pass is needed
    "fwd+bwd", # to test if the implementation is correct when both forward and backward pass are needed
    "fwd+2bwd", # to test if the implementation causes grad accumulation when multiple backward passes are needed
    "fwd+2bwd+rg", # to test if the implementation causes grad accumulation or change any of the input tensors' requires_grad flag when multiple backward passes are needed
]

# define configs for each score test case
score_test_cases = {
    "task1": {
        "case1": {
            "score": 20,
            "test_mode": "fwd",
            "b": 4,
            "s": 256,
            "h": 1024,
            "nh": 8,
            "e": 2048,
            "top_p": 0.7,
            "top_k": 32,
            "device": "cpu",  # Default to CPU
            "dtype": torch.float32,
        },
        "case2": {
            "score": 20,
            "test_mode": "fwd+bwd",
            "b": 1,
            "s": 256,
            "h": 512,
            "nh": 32,
            "e": 1024,
            "top_p": 0.5,
            "top_k": 50,
            "device": "cpu",  # Default to CPU
            "dtype": torch.bfloat16,
        },
        "case3": {
            "score": 20,
            "test_mode": "fwd+bwd",
            "b": 4,
            "s": 512,
            "h": 1024,
            "nh": 1,
            "e": 1024,
            "top_p": 0.95,
            "top_k": 5,
            "device": "cpu",  # Default to CPU
            "dtype": torch.float16,
        },
        "case4": {
            "score": 20,
            "test_mode": "fwd+2bwd",
            "b": 1,
            "s": 1024,
            "h": 1024,
            "nh": 8,
            "e": 2048,
            "top_p": 0.0,
            "top_k": 10,
            "device": "cpu",  # Default to CPU
            "dtype": torch.bfloat16,
        },
        "case5": {
            "score": 20,
            "test_mode": "fwd+2bwd+rg",
            "b": 2,
            "s": 1024,
            "h": 256,
            "nh": 32,
            "e": 512,
            "top_p": 0.8,
            "top_k": None,
            "device": "cpu",  # Default to CPU
            "dtype": torch.float32,
        }
    }
}

# Helper function to automatically set device
def get_device(device_type="cpu"):
    if torch.cuda.is_available() and device_type == "cuda":
        return torch.device("cuda")
    else:
        return torch.device("cpu")

@pytest.mark.parametrize(
    "case_key, case_config",
    score_test_cases['task1'].items(),
)
def test_task1(case_key: str, case_config: dict):
    # define hyper parameters
    test_mode = case_config["test_mode"]
    assert test_mode in test_modes, f"test_mode must be one of {test_modes}"
    
    b, s, h, nh, e = case_config["b"], case_config["s"], case_config["h"], case_config["nh"], case_config["e"]
    top_p, top_k = case_config["top_p"], case_config["top_k"]
    device, dtype = case_config["device"], case_config["dtype"]
    seed = case_config.pop("seed", SEED)
    atol, rtol = case_config.pop("atol", ATOL), case_config.pop("rtol", RTOL)

    # Get device (CPU or CUDA if available)
    device = get_device(device)

    # construct the input tensors
    torch.manual_seed(seed)
    input = torch.randn(b, s, h, device=device, dtype=dtype)
    weight = torch.randn(h, e, device=device, dtype=dtype)
    probs = torch.rand(b, s, device=device, dtype=dtype)
    
    # construct the reference output tensors
    output_ref, grad_input_ref, grad_weight_ref = matmul_with_importance_ref(
        input=input, 
        weight=weight, 
        probs=probs,
        num_heads=nh,
        top_p=top_p, 
        top_k=top_k
    )
    if "bwd" in test_mode:
        torch.manual_seed(seed)
        grad_output = torch.randn(output_ref.size(), dtype=output_ref.dtype, device=output_ref.device)
        output_ref, grad_input_ref, grad_weight_ref = matmul_with_importance_ref(
            input=input, 
            weight=weight, 
            probs=probs,
            grad_output=grad_output,
            num_heads=nh,
            top_p=top_p, 
            top_k=top_k
        )
    else:
        grad_output = None

    # get the output tensors to test from calling student's function (maybe multiple times)
    call_times = 2 if "2bwd" in test_mode else 1
    output, grad_input, grad_weight = None, None, None
    for _ in range(call_times):
        output, grad_input, grad_weight = matmul_with_importance(
            input=input, 
            weight=weight, 
            probs=probs,
            grad_output=grad_output,
            num_heads=nh,
            top_p=top_p, 
            top_k=top_k
        )
            
    # check if the output tensor is correct
    assert_close(output, output_ref, atol=atol, rtol=rtol)
    # check if the grad_input tensor is correct
    assert_close(grad_input, grad_input_ref, atol=atol, rtol=rtol)
    # check if the grad_weight tensor is correct
    assert_close(grad_weight, grad_weight_ref, atol=atol, rtol=rtol)
    
    if "rg" in test_mode:
        # check if the requires_grad flag remains False
        assert input.requires_grad == weight.requires_grad == probs.requires_grad == False, "Some internal attributes of certain input tensors are changed due to some side-effects."


def main():
    capture = ResultCapture()

    # Run pytest tests
    pytest.main(
        [
            '--quiet',
            '--tb=no', 
            '--disable-warnings',
            os.path.abspath(__file__)  # Run only this test file
        ], 
        plugins=[capture]
    )

    # Calculate and return the final score
    total_score = score_results(
        capture=capture,
        score_test_cases=score_test_cases,
        student_repo_path=student_repo_path,
    )

    # Optionally, write the result to a file or do something else with the score
    print(f"score:{total_score}")

if __name__ == "__main__":
    main()
