0. 测试者需要一个属于自己的能push的仓库，我会附上github仓库链接，git clone后根据上述2-4设置好webhook后就可以git push了，如果正常的话，你启动webhook_server.py的终端可以看到测试输出，然后你的github 仓库网页端，score-feedback分支可以在score.md中看到分数。
0. 测试者请自行访问a0仓库的readme.md去配置对应的python环境
1. 本方法使用的是github 提供的webhook功能，即用户向自己仓库push时，仓库会根据设置的webhook链接发送信号，从而让测试端能主动pull用户的最新版本代码，执行测试后，返回分数
2. webhook地址需学生在自己的仓库，点击setting，选择Webhooks,点击Add webhook，content type里面选择application/json,注意选上just the push event
3. Payload URL里面，需要填一个公网IP，比如说 https://baidu.com/webhook，前面是公网IP，后面是我们本地服务监听的接口，/webhook不能少。
4. webhook只能向公网ip发送信号，笔者是使用了ngrok，可以免费下载，支持linux环境，然后在终端输入 ngrok http port，其中port是webhook_server.py中webhook服务监听的端口，默认是5000,可以修改，如果使用默认值，就输入ngrok http 5000。
5. ngrok需要访问官网注册账号，然后 ngrok config add-authtoken <your auntoekn>， 详情请访问官方查询，包括安装 https://ngrok.com/downloads/linux
6. 仓库设置好webhook后，进入webhook文件夹，python3 webhook_server.py 就可以启动自动测试的脚本。
   
   如果amd方的服务器能直接访问公网，或者对ngrok的使用有所限制，那么请测试方自行修改webhook_server.py。




a0连接：
1. 建议git clone 后
2. git remote remove origin
3. git remote add origin ……你的新建仓库
4. git push -u origin --all
5. git push -u origin --tags