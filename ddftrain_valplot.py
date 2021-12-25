##!/usr/bin/python
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import seaborn as sns
import datetime
from datetime import datetime
import warnings
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
from sklearn.metrics import confusion_matrix, precision_score, accuracy_score,recall_score, f1_score,roc_auc_score
warnings.filterwarnings("ignore")
import time
import datetime
import re
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
import math
# from pyecharts.charts import *  
from pyecharts import options as opts  
from sklearn.tree import export_graphviz
from six import StringIO
from ddftrain_getparam import datadb_conn_param,read_param,sql_passwd,datadb_conn_data,log_return
from rating_card import woe_rule,generate_scorecard


###################验证结果输出#################
# '''
# * ks值：描述模型的正负样本的区分能力，越大值的箱越靠前代表模型效果越好。
#         假设该值不是严格递减，出现波动，越靠前波动说明模型排序能力越差。
# * 捕获率为当前负样本累计个数除以样本中负样本总数。用于衡量抓取负样本的能力。期望在阈值一下捕获率越大越好。
# '''
def ana_ks(model,X_test,y_test):
    '''
    目的：输出模型结果报告
    '''
    # model = lr_model  
    row_num, col_num = 0, 0  
    bins = 20  
    Y_predict = [s[1] for s in model.predict_proba(X_test)]  
    Y = y_test#.tolist()  
    nrows = Y.shape[0]    
    lis = [(Y_predict[i], Y.iloc[i]) for i in range(nrows)]  
    ks_lis = sorted(lis, key=lambda x: x[0], reverse=True)  
    bin_num = int(nrows/bins+1)  
    bad = sum([1 for (p, y) in ks_lis if y > 0.5])  
    good = sum([1 for (p, y) in ks_lis if y <= 0.5])  
    bad_cnt, good_cnt = 0, 0  
    KS = []  
    BAD = []  
    GOOD = []  
    BAD_CNT = []  
    GOOD_CNT = []  
    BAD_PCTG = []  
    BADRATE = []  
    dct_report = {}  
    for j in range(bins-1):  
        ds = ks_lis[j*bin_num: min((j+1)*bin_num, nrows)]  
        bad1 = sum([1 for (p, y) in ds if y > 0.5])  
        good1 = sum([1 for (p, y) in ds if y <= 0.5])  
        bad_cnt += bad1  
        good_cnt += good1  
        bad_pctg = round(bad_cnt/sum(y_test),3)  
        badrate = round(bad1/(bad1+good1),3)  
        ks = round(math.fabs((bad_cnt / bad) - (good_cnt / good)),3)  
        KS.append(ks)  
        BAD.append(bad1)  
        GOOD.append(good1)  
        BAD_CNT.append(bad_cnt)  
        GOOD_CNT.append(good_cnt)  
        BAD_PCTG.append(bad_pctg)  
        BADRATE.append(badrate)  
        dct_report['KS'] = KS  
        dct_report['负样本个数'] = BAD  
        dct_report['正样本个数'] = GOOD  
        dct_report['负样本累计个数'] = BAD_CNT  
        dct_report['正样本累计个数'] = GOOD_CNT  
        dct_report['捕获率'] = BAD_PCTG  
        dct_report['负样本占比'] = BADRATE  
    val_repot = pd.DataFrame(dct_report)  
#     print(val_repot)
    return val_repot
#调用
# val_repot = ana_1(model,X_test,y_test)

def ana_2(val_repot,title1):
    
    '''
    目的：ks可视化
    '''
    mpl.rcParams['font.sans-serif'] = ['SimHei']  
    np.set_printoptions(suppress=True)  
    pd.set_option('display.unicode.ambiguous_as_wide', True)  
    pd.set_option('display.unicode.east_asian_width', True)  
    line = (  

        Line()  
        .add_xaxis(list(val_repot.index))  
        .add_yaxis(  
            "分组负样本占比",  
            list(val_repot['负样本占比']),  
            yaxis_index=0,  
            color="red",  
        )  
        .set_global_opts(  
            title_opts=opts.TitleOpts(title=title1),  
        )  
        .extend_axis(  
            yaxis=opts.AxisOpts(  
                name="累计负样本占比",  
                type_="value",  
                min_=0,  
                max_=1,  
                position="right",  
                axisline_opts=opts.AxisLineOpts(  
                    linestyle_opts=opts.LineStyleOpts(color="red")  
                ),  
                axislabel_opts=opts.LabelOpts(formatter="{value}"),  
            )  

        )  
        .add_xaxis(list(val_repot.index))  
        .add_yaxis(  
            "KS",  
            list(val_repot['KS']),  
            yaxis_index=1,  
            color="blue",  
            label_opts=opts.LabelOpts(is_show=False),  
        )  
    )
    return line
#调用
# line= ana_2(val_repot)
# line.render_notebook()

def auc_confusion(X_test,y_test):
    '''
    目的：准确率、混淆矩阵计算
    '''
    accuracy = accuracy_score(model.predict(X_test),y_test)
    print("accuarcy: %.2f%%" % (accuracy*100.0))
    print("auc: %.4f" % (roc_auc_score(model.predict(X_test),y_test)))
    y_true = y_test.copy()
    y_pred = model.predict(X_test)
    C2= confusion_matrix(y_true,y_pred, labels=[0, 1])
    print(C2)
    sns.heatmap(C2,annot=True)

def auc_r(model,X_test,y_test):
    '''
    目的：混淆矩阵可视化
    '''
    fpr, tpr, thresholds  =  roc_curve(y_test,model.predict(X_test)) 
    roc_auc =auc(fpr, tpr) 
    plt.figure()
    lw = 2
    np.set_printoptions(suppress=True)
    plt.figure(figsize=(5,5))
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc) ###假正率为横坐标，真正率为纵坐标做曲线
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

def remove_prefix(str, prefix):
    return str.lstrip(prefix)

def importance_fea(model,X_train,algorithm,model_type):
    ''''
    目的：特征重要度数值输出
    model：模型
    X_train：训练集
    algorithm：选择的算法
    '''
    if algorithm == 12:
        #逻辑回归模型
        importances = abs(model.coef_)
        n = importances.shape[1]
        importances = importances.reshape(n,)
    else:
    #树模型
        importances = model.feature_importances_
    # 获取特征名称
    feat_names = X_train.columns
    if model_type==1:
        columms_new = []
        for i in feat_names:
            fea = remove_prefix(i, 'woe_bin_')
            columms_new.append(fea)
        feat_names = columms_new
    fea_impor = pd.DataFrame({'fea_name':feat_names,'importances':importances})
    fea_impor = fea_impor[abs(fea_impor['importances'])>0].sort_values(['importances'],ascending=False)
    # 排序
    indices = np.argsort(importances)[::-1]
    return fea_impor

def card_rule_reslut(model, df_woe, feature_cols,model_id,train_id):
    '''
    目的：评分卡规则生成
    '''
    #评分卡
    score_card = generate_scorecard(model.coef_, df_woe, feature_cols)
    score_card.columns = ['variable','binning','score']
    score_card['model_info_id'] = model_id
    score_card['training_id'] = train_id
    score_card['binning'] = score_card['binning'].astype('string')
    #score_card['variable'] = score_card['variable'].astype('string')
    #评分卡
    engine,cursor,conn = datadb_conn_param() ####完成替换
    pd.io.sql.to_sql(score_card,'ddf_model_predict_score_card', engine,index=False, schema='dev_ddframework', if_exists='append')  #fail 、replace
    return ('完成评分卡规则输出')

def result_visualization(model,X_train,X_test,y_test,algorithm,model_id,train_id,model_type):
    '''
    目的：ddf所需结果输出
    '''
#结果可视化
    #准确率
    accuracy = accuracy_score(model.predict(X_test),y_test)
    #混淆矩阵
    C2= confusion_matrix(y_test,model.predict(X_test), labels=[0, 1])
    tp = C2[1][1]#真阳
    fp = C2[0][1]#伪阳
    fn = C2[1][0]#伪阴
    tn = C2[0][0]#真阴
    #roc曲线
    ###假正率（fpr）为横坐标，真正率（tpr）为纵坐标做曲线
    fpr, tpr, thresholds  = roc_curve(y_test,model.predict(X_test))
    roc_result = pd.DataFrame(columns=['model_info_id','training_id','fpr','tpr','remark'])
    auc = roc_auc_score(model.predict(X_test),y_test)
    roc_result['fpr'] = fpr
    roc_result['tpr'] = tpr
    roc_result['remark'] = auc
    roc_result['model_info_id'] = model_id
    roc_result['training_id'] = train_id
    #特征重要系
    fea_impor = importance_fea(model,X_train,algorithm,model_type)
    fea_impor.columns = ['feature','importancy']
    fea_impor['model_info_id'] = model_id
    fea_impor['training_id'] = train_id
    #模型报告
    val_repot = ana_ks(model,X_test,y_test)
    val_repot.columns = ['ks_value','negative_sample_num','positive_sample_num','negative_sample_total_num',\
                        'positive_sample_total_num','capture_rate','negative_sample_percentage']
    val_repot['model_info_id'] = model_id
    val_repot['training_id'] = train_id
    #连接数据库
    engine,cursor,conn = datadb_conn_param() ####完成替换
    #准确率插入数据库
    cursor.execute('insert into ddf_model_predict_accuracy_rate(model_info_id, training_id, accuracy_rate) values(%s, %s, %s)' % (model_id, train_id, accuracy))
    #混淆矩阵插入数据库
    cursor.execute('insert into ddf_model_predict_confusion_matrix(model_info_id, training_id, tp,fp,fn,tn) values(%s, %s, %s, %s, %s, %s)' % (model_id, train_id, tp,fp,fn,tn))
    conn.commit()
    #模型报告
    pd.io.sql.to_sql(val_repot,'ddf_model_predict_report', engine,index=False, schema='dev_ddframework', if_exists='append')  #fail 、replace
    #roc
    pd.io.sql.to_sql(roc_result,'ddf_model_predict_roc', engine,index=False, schema='dev_ddframework', if_exists='append')  #fail 、replace
    #特征重要度输出
    pd.io.sql.to_sql(fea_impor,'ddf_model_predict_feature_importancy', engine,index=False, schema='dev_ddframework', if_exists='append')  #fail 、replace
    #基本信息模型状态更新
    cursor.execute('update ddf_model_info set model_status=%s where model_info_id=%s', [2,int(model_id)])
    #训练模型状态更新
    cursor.execute('update ddf_model_training_history set result_flag=%s where id=%s', [2,int(train_id)])
    cursor.execute('commit')
    cursor.close()
    conn.close()
    title_6 = '模型验证'
    tip_6 = '完成验证结果数据输出，程序结束'
    return title_6,tip_6


# def tree_plot(X_train,model):
#     # 需安装GraphViz和pydotplus进行决策树的可视化
#     # 特征向量
#     feature_names = X_train.columns
#     # 文件缓存
#     dot_data = StringIO()
#     # 将决策树导入到dot中
#     export_graphviz(model, out_file=dot_data,  
#                     filled=True, rounded=True,
#                     special_characters=True,feature_names = feature_names,class_names=['0','1','2'])
#     # 将生成的dot文件生成graph
#     graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
#     # 将结果存入到png文件中
#     graph.write_png('diabetes_no.png')
#     # 显示
# #     Image(graph.create_png())
#     return graph

#调用
# graph = tree_plot(X_train,model)
# Image(graph.create_png())

# def fea_affect(X_test):
#     '''
#     目的：模型可解释性affect
#     '''
#     fea_list = X_test.columns.tolist()
#     for i in fea_list:
#         base_features = X_test.columns.values.tolist()
#         # base_features.remove('target')#delete the target from the list
#         feat_name = i
#         pdp_dist = pdp.pdp_isolate(model=model, dataset=X_test, model_features=base_features, feature=feat_name)
#         pdp.pdp_plot(pdp_dist, feat_name)
#         plt.show()
#调用
# fea_affect(X_test)

# def priori_probability(train_h,label):
#     '''
#     目的：模型可解释性priori
#     '''
#     fea_list = train_h.columns.tolist()
#     fea_list.remove(label)
#     fea_list.remove('creditlevel')
#     for i in fea_list:
#         fig, axes, summary_df = info_plots.target_plot(\
#             df=train_h, feature=i, feature_name=i, target=label, show_percentile=True)
#调用
# fea_affect(train_h,label)

# def fea_importance(model,X_train):
#     ''''特征重要度可视化'''
#     # 获取特征重要性
#     importances = model.feature_importances_
#     # 获取特征名称
#     feat_names = X_train.columns
#     fea_impor = pd.DataFrame({'fea_name':feat_names,'importances':importances})
#     print(fea_impor[fea_impor['importances']>0].sort_values(['importances'],ascending=False))
#     # 排序
#     indices = np.argsort(importances)[::-1]
#     # 绘图
#     myfont = matplotlib.font_manager.FontProperties(fname='/root/anaconda3/lib/python3.6/site-packages/matplotlib/mpl-data/fonts/ttf/simhei.ttf')
#     plt.figure(figsize=(12,6))
#     plt.title("Feature importances by Decision Tree")
#     plt.bar(range(len(indices)), importances[indices], color='lightblue',  align="center")
#     plt.step(range(len(indices)), np.cumsum(importances[indices]), where='mid', label='Cumulative')
#     plt.xticks(range(len(indices)), feat_names[indices], rotation='vertical',fontsize=14,fontproperties=myfont)
#     plt.xlim([-1, len(indices)])
#     plt.show()

# def get_numerical_serial_fea(data,feas):
#     '''
#     目的：连续型、离散型特征可视化
#     '''
#     numerical_serial_fea = []  #连续性
#     numerical_noserial_fea = [] #离散型
#     for fea in feas:
#         temp = data[fea].nunique()
#         if temp <= 10:
#             numerical_noserial_fea.append(fea)
#             continue
#         numerical_serial_fea.append(fea)
#     return numerical_serial_fea,numerical_noserial_fea
# def plot(data_train,fea):
#     ##变量分类
#     numerical_fea = list(data_train.select_dtypes(exclude=['object']).columns)
#     category_fea = list(filter(lambda x: x not in numerical_fea,list(data_train.columns)))
#     numerical_serial_fea,numerical_noserial_fea = get_numerical_serial_fea(data_train,numerical_fea)
#     print(numerical_serial_fea)
#     print(numerical_noserial_fea)
#     numerical_noserial_fea.remove(fea)
#     #类别数据可视化
#     for i in range(0,len(numerical_noserial_fea),2):
#         f, [ax1,ax2] = plt.subplots(1, 2, figsize=(20, 5))
#         sns.countplot(x=numerical_noserial_fea[i], hue=fea, data=data_train,ax=ax1)
#         sns.countplot(x=numerical_noserial_fea[i+1], hue=fea, data=data_train,ax=ax2)
#     #数值型变量可视化
#     ##注意第一个参数不是0就是1
#     for i in range(0,len(numerical_serial_fea)):
#         f, [ax1,ax2] = plt.subplots(1, 2, figsize=(20, 5))
#         #         data_train[numerical_serial_fea[i]].hist(ax=ax1)
#         #         data_train[numerical_serial_fea[i+1]].hist(ax=ax2)
#         data_train[data_train[fea] == 1][numerical_serial_fea[i]].plot(kind='hist',bins=100,color='r',ax= ax1)
#         data_train[data_train[fea] == 0][numerical_serial_fea[i]].plot(kind='hist',bins=100,color='r',ax= ax2)
#调用       
#plot(data_train,'punish')

# def corr_of_label(df,label):
#     '''
#     目的：相关性可视化
#     '''
#     corrmat = df.corr()
#     top_corr_features = corrmat.index[abs(corrmat[label])>0.01]
#     plt.figure(figsize=(20,10))
#     colormap=plt.cm.RdBu
#     g = sns.heatmap(df[top_corr_features].corr(),annot=True,cmap=colormap)