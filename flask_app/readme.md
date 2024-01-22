# 文件介绍

- pika文件夹放了使用rabbit MQ队列的发、收文件

     [rabbit MQ 介绍](https://www.cnblogs.com/guyuyun/p/14970592.html)

- flask_client_thread.py
     
     用flask做后端，读取本地文件夹results/5 的所有图像，然后拼接保存到本地文件夹static/image 推到前端显示。
     缺点：拼接耗时，时间太长时网页容易失去相应。不能拼接太多的图片，只能小数据量处理。
