from flask import Flask, request, jsonify
import subprocess
import os

app = Flask(__name__)
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
    subprocess.Popen(['bash', './a0/test_script.sh', ssh_url, repo_name,student_name])
    return jsonify({'status': 'finish test'}), 200



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000,debug=True)
