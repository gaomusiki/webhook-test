import sys
import os

import pandas as pd

SCORE_FEEDBACK_FILENAME = "score.md"

student_repo_root = os.getenv("STUDENT_REPO_ROOT", None)
if student_repo_root is None:
    print("env variable `STUDENT_REPO_ROOT` is not set")
    sys.exit(1)
    
assignment_name = os.getenv("ASSIGNMENT", None)
if assignment_name is None:
    print("env variable `ASSIGNMENT` is not set")
    sys.exit(1)


def read_total_score(score_table: str) -> int:
    total_score_line = score_table.split("\n")[-3]
    total_score = int(total_score_line.split("|")[2].strip())
    return total_score


def main():
    score_dict = {
        "account": [],
        "score": [],
    }
    for idx, student_repo in enumerate(os.listdir(student_repo_root)):
        if not os.path.isdir(os.path.join(student_repo_root, student_repo)):
            continue
        if not student_repo.startswith(assignment_name):
            continue
        
        student_account = student_repo[len(assignment_name) + 1:]
        
        student_score_file = os.path.join(
            student_repo_root, 
            student_repo,
            SCORE_FEEDBACK_FILENAME
        )
        if not os.path.exists(student_score_file):
            continue
        
        with open(student_score_file, "r") as f:
            score_table = f.read()

        student_total_score = read_total_score(score_table)
        print(f"{str(idx+1)+'.':5s} {student_account:20s} got the total score: \t{student_total_score}\t for {assignment_name}")
        score_dict["account"].append(student_account)
        score_dict["score"].append(student_total_score)

    score_df = pd.DataFrame(score_dict).sort_values(by="score", ascending=False)
    average_score = round(score_df['score'].mean(), 2)
    score_df.loc['AVERAGE'] = ['AVERAGE', average_score]
    score_df.to_csv(f"{assignment_name}_score.csv", index=False)


if __name__ == "__main__":
    main()