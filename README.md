# DDF机器学习

* requirements.txt为py环境文件，使用命令pip freeze > requirements.txt生成依赖库及其版本。使用命令pip install -r requirements.txt （在该文件所在目录执行，或在命令中写全文件的路径），就能自动把所有的依赖库给装上
* 训练只需运行run.py文件，如在run.py所在路径下python run.py
* 预测只需运行predict_run.py文件，如在run.py所在路径下python run.py

## ddf_train文件夹
包含了9个py文件，分别为
ddftrain_getparam：获取参数配置文件
ddftrain_readdata：读取数据文件
ddftrain_dataprosesing：数据处理文件
ddftrain_featureenginneer：特征工程文件
ddftrain_datasplit：数据划分文件
ddftrain_modelchoose：算法选择文件
ddftrain_valplot：验证结果输出文件
rating_card：评分卡文件
run：运行主文件


## ddf_predict文件夹
包含了3个py文件，分别为
predict_data_model：配置、数据集、模型准备文件
rating_card：评分卡文件
predict_run：运行主文件
