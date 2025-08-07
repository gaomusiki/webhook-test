
#!/bin/bash

SEP="========================================================================"
SUBMITS_ROOT="submits"
CLASSROOM="nju-llm-course-classroom"

export ASSIGNMENT="assignment-4-modeling-transformer"

export STUDENT_REPO_ROOT="$SUBMITS_ROOT/$ASSIGNMENT-submissions"

echo "$SEP"
echo "Statistics students' score for $ASSIGNMENT"
echo "$SEP"

python stats_score.py

echo "$SEP"
echo "Done!"
echo "$SEP"