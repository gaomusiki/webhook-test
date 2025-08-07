from flask import Flask, request, jsonify
import subprocess
import os
import sys

app = Flask(__name__)

def get_assignment_id():
    """从命令行参数中获取assignment_id"""
    for arg in sys.argv[1:]:
        if arg.startswith('assignment_id='):
            return arg.split('=')[1]
    return 'a4'  # 默认值

# 在应用启动时获取assignment_id
assignment_id = get_assignment_id()


@app.route('/webhook', methods=['POST'])
def github_webhook():
    # 获取 GitHub 传过来的 JSON 数据
    data = request.json
    #print(data)  # 打印接收到的 JSON 数据，帮助调试
    
    ref = data.get('ref',None)
    
    # 如果不是推送到 main 分支，直接返回
    if ref != "refs/heads/main" and ref!=None:
        print(f"Push to non-main branch: {ref}. Skipping webhook processing.")
        return jsonify({'status': 'skipped'}), 200
    

    ssh_url = data['repository']['ssh_url']
    #ssh_url = repo_url.replace("https://github.com/", "git@github.com:").replace("https://", "").replace("http://", "")
    print(ssh_url)
    repo_name = data['repository']['name']
    # 启动后台线程进行拉取代码和测试
    print(repo_name)
    student_name = repo_name.split('-')[-1]
    print(student_name)
    # 使用动态获取的assignment_id构建脚本路径
    script_path = f'./{assignment_id}/test_script.sh'
    subprocess.Popen(['bash', script_path, ssh_url, repo_name, student_name])
    
    return jsonify({'status': 'finish test'}), 200



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000,debug=True)
