import os
import pandas as pd
import numpy as np
import time
import datetime
import missingno as msno
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif']=['SiHei']
matplotlib.rcParams['axes.unicode_minus']=False


def data_read(data_path, file_name):
    '''
    获取数据集
    将标签有1,2转成0,1
    0表示好用户，1表示坏用户
    '''
    df = pd.read_csv(os.path.join(data_path, file_name), delim_whitespace=True, header=None)
     ##变量重命名
    columns = ['status_account','duration','credit_history','purpose', 'amount',
               'svaing_account', 'present_emp', 'income_rate', 'personal_status',
               'other_debtors', 'residence_info', 'property', 'age',
               'inst_plans', 'housing', 'num_credits',
               'job', 'dependents', 'telephone', 'foreign_worker', 'target']
    df.columns = columns
    df.target = df.target - 1
    
    return df


# path = 'D:\\Repositories\\financial_big_data_risk_control_modeling_practice\\chapter4'
path = ''
file_name = 'german.csv'

df = data_read(path, file_name)


df.head()


def category_continue_separation(df, feature_names):
    '''
    将数据列以连续变量和非连续变量分开
    '''
    category_var = []
    numerical_var = []
    if 'target' in feature_names:
        feature_names.remove('target')
    
    numerical_var = list(df[feature_names].select_dtypes(include=['int','float','int32','float32','int64','float64']).columns.values)
    category_var = [x for x in feature_names if x not in numerical_var]
    
    return category_var, numerical_var


feature_names = list(df.columns)


feature_names


feature_names.remove('target')


# 分开连续变量和非连续变量
categorical_var, numerical_var = category_continue_separation(df,feature_names)


df.describe()


def add_str(x):
    '''
    为x随机添加一个字符
    '''
    str_1 = ['get_ipython().run_line_magic("','", " ','/t','$',';','@']")
    str_2 = str_1[np.random.randint(0,high=len(str_1)-1)]
    
    return x + str_2


# 注入“脏数据”
df.status_account = df.status_account.apply(add_str)


df.status_account


def add_time(num, style='get_ipython().run_line_magic("Y-%m-%d'):", "")
    '''
    添加num个格式为style的时间字符串
    '''
    # 生成时间戳
    start_time = time.mktime((2010,1,1,0,0,0,0,0,0))
    stop_time = time.mktime((2015,1,1,0,0,0,0,0,0))
    re_time = []
    
    for i in range(num):
        rand_time = np.random.randint(start_time, stop_time)
        # 转成时间元组
        date_touple = time.localtime(rand_time)
        re_time.append(time.strftime(style, date_touple))
        
    return re_time


# 添加两列时间格式的数据
df['apply_time'] = add_time(df.shape[0], 'get_ipython().run_line_magic("Y-%m-%d')", "")
df['job_time'] = add_time(df.shape[0], "get_ipython().run_line_magic("Y/%m/%d", " %H:%M:%S\")")


df['apply_time']


df['job_time']


def add_row(df_temp, num):
    '''
    从df_temp中随机取num行数据
    '''
    index_1 = np.random.randint(low=0,high=df_temp.shape[0]-1, size = num)
    return df_temp.loc[index_1]


# 添加冗余数据
df_temp = add_row(df, 10)
df = pd.concat([df, df_temp],axis=0,ignore_index=True)


df.shape


# 设置head（）方法显示多列或者显示全部
pd.set_option('display.max_columns', 10)
df.head()


pd.set_option('display.max_columns', None)
df.head()


# 离散变量看下范围
df.status_account.unique()


# 特殊字符清洗
df.status_account = df.status_account.apply(lambda x: x.replace(' ','').replace('get_ipython().run_line_magic("','').replace('/t','').replace('$','').replace('@','').replace(';',''))", "")


df.status_account.unique()


# 时间格式统一
# 统一为'get_ipython().run_line_magic("Y-%m-%d'", "")
# df['job_time'] = df['job_time'].apply(lambda x:datetime.datetime.strptime( x, 'get_ipython().run_line_magic("d/%m/%Y", " %H:%M:%S.%f))")
df['job_time'] = add_time(df.shape[0], 'get_ipython().run_line_magic("Y-%m-%d')", "")


df['apply_time'] = df['apply_time'].apply(lambda x:datetime.datetime.strptime( x, 'get_ipython().run_line_magic("Y-%m-%d'))", "")


df['apply_time']


# 样本去冗余
df.drop_duplicates(subset=None, keep='first',inplace=True)


df.shape


# 按照订单去冗余
df['order_id'] = np.random.randint(low=0,high=df.shape[0]-1,size=df.shape[0])
df.drop_duplicates(subset=['order_id'],keep='first',inplace=True)


df.shape


df[numerical_var].describe()


df.reset_index(drop=True, inplace=True)


var_name = categorical_var + numerical_var


for i in var_name:
    num = np.random.randint(low=0,high = df.shape[0]-1)
    index_1 = np.random.randint(low = 0, high = df.shape[0] - 1, size=num)
    index_1 = np.unique(index_1)
    df[i].loc[index_1] = np.nan


# 缺失值绘图
msno.bar(df, labels=True, figsize=(10,6),fontsize=10)


#对于连续数据绘制箱线图，观察是否有异常值 
plt.figure(figsize=(10,6)) # 设置图形尺寸大小
for j in range(1, len(numerical_var)+1):
    plt.subplot(2,4,j)
    df_temp = df[numerical_var[j-1]][~df[numerical_var[j-1]].isnull()]
    plt.boxplot(df_temp
                ,notch = False # 中位线处不设置凹陷
                ,widths=0.2    # 设置箱体宽度
                ,medianprops={'color':'red'}  # 中位线设置为红色
                ,boxprops=dict(color="blue")  # 箱体边框设置为蓝色
                ,labels=[numerical_var[j-1]]  # 设置标签
                ,whiskerprops={'color':'black'} # 设置须的颜色，黑色
                ,capprops={'color':'green'}     # 设置箱线图顶端和末端横线的属性，颜色为绿色
                ,flierprops={'color':'purple','markeredgecolor':'purple'} # 异常值属性
               )
    plt.show()


# 连续变量不同类别下的分布
for i in numerical_var:
    df_temp = df.loc[~df[i].isnull(),[i,'target']] # 第一个为条件，第二个为要取的列
    df_good = df_temp[df_temp.target == 0]
    df_bad = df_temp[df_temp.target == 1]
    # 计算统计量
    valid = round(df_temp.shape[0] / df.shape[0]*100,2)
    Mean = round(df_temp[i].mean(),2)
    Std = round(df_temp[i].std(), 2)
    Max = round(df_temp[i].max(), 2)
    Min = round(df_temp[i].min(), 2)
    
    # 绘图
    plt.figure(figsize=(10,6))
    fontsize_1 = 12
    plt.hist(df_good[i], bins=20, alpha=0.5,label='好样本')
    plt.hist(df_bad[i], bins=20, alpha=0.5, label='坏样本')
    plt.ylabel(i, fontsize=fontsize_1)
    plt.title(f'valid={valid},Mean={Mean},Std={Std},Max={Max},Min={Min}')
    plt.legend()
    file = f"{i}.png"
    plt.savefig(file)
    plt.close(1)


# 离散变量不同类别下的分布
for i in categorical_var:
    df_temp = df.loc[~df[i].isnull(),[i,'target']]
    df_bad = df_temp[df_temp.target == 1]
    valid = round(df_temp.shape[0] / df.shape[0] * 100, 2)
    
    bad_rate = []
    bin_rate = []
    var_name = []
    
    for j in df[i].unique():
        if pd.isnull(j):
            df_1 = df[df[i].isnull()]
            bad_rate.append(sum(df_1.target) / df_1.shape[0])
            bin_rate.append(df_1.shape[0] / df.shape[0])
            var_name.append('NA')
        else:
            df_1 = df[df[i] == j]
            bad_rate.append(sum(df_1.target) / df_1.shape[0])
            bin_rate.append(df_1.shape[0] / df.shape[0])
            var_name.append(j)
            
    df_2 = pd.DataFrame({'var_name':var_name, 'bin_rate':bin_rate,'bad_rate':bad_rate})
    plt.figure(figsize=(10,6))        
    fontsize_1 = 12
    plt.bar(np.arange(1,df_2.shape[0]+1),df_2.bin_rate,0.1,color='black',alpha=0.5,label='占比')
    plt.xticks(np.arange(1,df_2.shape[0]+1),df_2.var_name)
    plt.plot(np.arange(1,df_2.shape[0]+1),df_2.bad_rate,color='green',alpha=0.5,label='坏样本率')
    plt.ylabel(i,fontsize=fontsize_1)
    plt.title(f'valid rate={valid}get_ipython().run_line_magic("')", "")
    plt.legend()
    file = f"{i}.png"
    plt.savefig(file)
    plt.close(1)



