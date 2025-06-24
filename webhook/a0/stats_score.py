import sys
import os
import pandas as pd

SCORE_FEEDBACK_FILENAME = "score.md"

# 获取学生仓库路径和作业名称
student_repo_path = os.getenv("STUDENT_REPO_PATH", None)
if student_repo_path is None:
    print("env variable `STUDENT_REPO_PATH` is not set")
    sys.exit(1)


# 用于提取score.md中的总分数
def read_total_score(score_table: str) -> int:
    total_score_line = score_table.split("\n")[-3]  # 获取倒数第三行
    total_score = int(total_score_line.split("|")[2].strip())  # 从那一行提取分数
    return total_score

def main():
    # 这里假设student_repo_path是单个学生的仓库路径
    student_score_file = os.path.join(student_repo_path, SCORE_FEEDBACK_FILENAME)
    
    if not os.path.exists(student_score_file):
        print(f"{SCORE_FEEDBACK_FILENAME} does not exist in the specified repo path.")
        sys.exit(1)

    # 读取score.md文件
    with open(student_score_file, "r") as f:
        score_table = f.read()

    # 获取学生总分
    student_total_score = read_total_score(score_table)
    
    # 打印学生分数
    print(student_total_score)

if __name__ == "__main__":
    main()
