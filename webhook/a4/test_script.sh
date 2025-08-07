#!/bin/bash
SEP="========================================================================"
SCORE_FEEBACK_BRANCH_NAME="score-feedback"
STUDENT_REPO_ROOT="submits/"
cd "a4"
mkdir -p "${STUDENT_REPO_ROOT}"

# 获取传入的 repo_url 和 student_name
repo_url=$1
repo_name=$2
student_name=$3

# 初始化一些变量
if [ -z "$repo_url" ] || [ -z "$student_name" ]; then
    echo "Error: repo_url and student_name must be provided!"
    exit 1
fi

# 检查学生的仓库是否已存在
student_repo_path="${STUDENT_REPO_ROOT}${repo_name}"

if [ -d "$student_repo_path" ]; then
    echo "$SEP"
    echo "Repository for $student_name already exists, skipping clone."
    echo "$SEP"
else
    # 克隆学生的仓库
    echo "$SEP"
    echo "Cloning the repository: $repo_url..."
    echo "$SEP"
    start_time=$(date +%s)

    # 使用 Git 克隆仓库
    git clone "$repo_url" "${STUDENT_REPO_ROOT}${repo_name}" || exit 1
    end_time=$(date +%s)
    execution_time=$((end_time - start_time))
    minutes=$((execution_time / 60))
    seconds=$((execution_time % 60))

    echo ""; echo "Done in $minutes min $seconds sec."; echo ""
fi

# 进入仓库并执行测试
echo "$SEP"
echo "Testing for $student_name"
echo "$SEP"

cd "${student_repo_path}"

# 拉取最新代码
git checkout main && git pull origin main

# 执行测试脚本并获取分数
export STUDENT_REPO_PATH="$(pwd)"

cd -
python3 test_score.py  # 运行测试脚本并获取分数
score=$(python stats_score.py)

# 输出结果
echo "Score for $student_name: $score"


# 将结果推送到 score-feedback 分支
# 输出信息，调试
echo ""
echo "$SEP"
echo "Pushing score feedback for $student_repo_name"
echo "$SEP"

# 进入学生仓库路径
cd "${student_repo_path}"

# 确保在 main 分支上
git checkout main

# 拉取最新的 main 分支
git pull origin main

# 检查是否已经存在 score-feedback 分支
# 如果存在，则删除并重新创建
git checkout -b $SCORE_FEEBACK_BRANCH_NAME || \
( git branch -D $SCORE_FEEBACK_BRANCH_NAME && \
    (git push --force origin --delete $SCORE_FEEBACK_BRANCH_NAME || true) \
    && git checkout -b $SCORE_FEEBACK_BRANCH_NAME )

# 将 score.md 文件添加到 git，并推送到远程仓库
git add score.md
git commit -m "Done test and added score feedback"

# 推送到远程的 score-feedback 分支
git push -u origin $SCORE_FEEBACK_BRANCH_NAME

# 切换回 main 分支
git checkout main

# 返回之前的目录
cd -


# 记录或更新分数到文件 (txt 或 json)
score_file="student_scores.txt"

# 如果文件不存在，则创建
if [ ! -f "$score_file" ]; then
    touch "$score_file"
fi

# 检查学生是否已经存在，如果存在则更新分数，否则添加新记录
echo "Debug: student_name = $student_name, score = $score"

if grep -q "$student_name" "$score_file"; then
    # 更新分数，使用 | 作为分隔符，避免 / 引起的解析问题
    sed -i "s|^$student_name.*|$student_name $score|" "$score_file"
else
    # 如果学生记录不存在，添加新记录
    echo "$student_name $score" >> "$score_file"
fi


echo "$SEP"
echo "DONE! Processed student: $student_name"
echo "$SEP"
