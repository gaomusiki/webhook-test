import os
from datetime import datetime

import pandas as pd

import torch
from torch.testing import assert_close


# global test settings for all assignments
TOTAL_SCORE = 100
SCORE_FEEDBACK_FILENAME = "score.md"
ERROR_MSG_PREFIX_TO_CUTOFF = "E   "
MAX_ERROR_MSG_LENGTH = 200


class ResultCapture:
    def __init__(self):
        self.passed = []
        self.failed = []

    def pytest_runtest_logreport(self, report):
        if report.when == 'call':
            if report.passed:
                self.passed.append(report.nodeid)
            elif report.failed:
                self.failed.append({
                    'nodeid': report.nodeid,
                    'longrepr': str(report.longrepr),
                })

    def pytest_terminal_summary(self, terminalreporter, exitstatus):
        pass


def score_results(
    capture: ResultCapture, 
    score_test_cases: dict, 
    student_repo_path: str
) -> None:
    df_dict = {
        "Test Case": [],
        "Score": [],
        "Status": [],
        "Error Message": [],
    }
    
    total_score = 0
    total_passed_score = 0
    for task, cases in score_test_cases.items():
        for case_key, case_config in cases.items():
            # fill test case name
            df_dict['Test Case'].append(
                f"{task.capitalize()} - {case_key.capitalize()}"
            )
            
            # get test case score
            score = case_config['score']
            
            # init passed score
            passed_score = 0
            
            # search if this case for this task is passed
            # and get status and error message
            status, error_msg = None, ''
            for passed_case in capture.passed:
                if task in passed_case and case_key in passed_case:
                    status = "✅"
                    passed_score = score
                    
                    break
            else: # if not, search if this case for this task is failed
                for failed_case in capture.failed:
                    failed_case_name = failed_case['nodeid']
                    failed_case_msg = failed_case['longrepr']
                    
                    if task in failed_case_name and case_key in failed_case_name:
                        if "AssertionError" in failed_case_msg:
                            status = "❌"
                        elif "Failed: Timeout" in failed_case_msg:
                            status = "🕛"
                        else:
                            status = "❓"
                            
                        # process error message
                        if failed_case_msg.startswith(ERROR_MSG_PREFIX_TO_CUTOFF):
                            failed_case_msg = failed_case_msg[len(ERROR_MSG_PREFIX_TO_CUTOFF):]
                        if len(failed_case_msg) > MAX_ERROR_MSG_LENGTH:
                            failed_case_msg = f"{failed_case_msg[:MAX_ERROR_MSG_LENGTH//2]} ... {failed_case_msg[-MAX_ERROR_MSG_LENGTH//2:]}"
                        error_msg = failed_case_msg
                        
                        break
                else:
                    raise ValueError(f"Case {case_key} for task {task} is not found in results.")
            
            # fill passed score
            df_dict['Score'].append(passed_score)
            
            # fill status
            df_dict['Status'].append(status)
            
            # fill error message
            df_dict['Error Message'].append(error_msg)
            
            # accumulate score
            total_score += score
            total_passed_score += passed_score
    
    # fill score summary
    df_dict['Test Case'].append('Total')
    
    assert total_score == TOTAL_SCORE, "The total score should be 100."
    df_dict['Score'].append(total_passed_score)
    
    if total_passed_score == total_score:
        df_dict['Status'].append("😊")
    else:
        df_dict['Status'].append("🥺")
        
    df_dict['Error Message'].append('')
    
    # transform to dataframe to markdown
    df = pd.DataFrame(df_dict)
    score_md_str = df.to_markdown(index=False)
    current_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    score_md_str = f"# Score Summary\n\n{score_md_str}\n\n*This score feedback is generated at {current_time_str}*"
    
    # write score file
    score_file_path = os.path.join(student_repo_path, SCORE_FEEDBACK_FILENAME)
    with open(score_file_path, "w", encoding="utf-8") as f:
        f.write(score_md_str)


def check_if_io_meta_is_match(
    output: torch.Tensor,
    input: torch.Tensor,
) -> None:
    # check if the meta attribute of outout tensor is correct
    assert output.dtype == input.dtype, f"The dtype of output tensor should be {input.dtype}, but got {output.dtype}."
    assert output.device == input.device, f"The device of output tensor should be {input.device}, but got {output.device}."
    
    
def check_if_param_reset_is_fine(
    module: torch.nn.Module,
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> None:
    # check if the reset_parameters function works fine
    for name_old, param in module.named_parameters():
        param_old = param.clone()
        module.reset_parameters()
        for name, param in module.named_parameters():
            if name == name_old:
                assert_close(param_old, param, atol=atol, rtol=rtol, msg=f"The parameter `{name}` is not reset correctly.")