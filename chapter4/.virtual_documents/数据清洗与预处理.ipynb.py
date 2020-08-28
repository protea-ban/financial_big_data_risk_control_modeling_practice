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
    ��ȡ���ݼ�
    ����ǩ��1,2ת��0,1
    0��ʾ���û���1��ʾ���û�
    '''
    df = pd.read_csv(os.path.join(data_path, file_name), delim_whitespace=True, header=None)
     ##����������
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
    �������������������ͷ����������ֿ�
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


# �ֿ����������ͷ���������
categorical_var, numerical_var = category_continue_separation(df,feature_names)


df.describe()


def add_str(x):
    '''
    Ϊx������һ���ַ�
    '''
    str_1 = ['get_ipython().run_line_magic("','", " ','/t','$',';','@']")
    str_2 = str_1[np.random.randint(0,high=len(str_1)-1)]
    
    return x + str_2


# ע�롰�����ݡ�
df.status_account = df.status_account.apply(add_str)


df.status_account


def add_time(num, style='get_ipython().run_line_magic("Y-%m-%d'):", "")
    '''
    ���num����ʽΪstyle��ʱ���ַ���
    '''
    # ����ʱ���
    start_time = time.mktime((2010,1,1,0,0,0,0,0,0))
    stop_time = time.mktime((2015,1,1,0,0,0,0,0,0))
    re_time = []
    
    for i in range(num):
        rand_time = np.random.randint(start_time, stop_time)
        # ת��ʱ��Ԫ��
        date_touple = time.localtime(rand_time)
        re_time.append(time.strftime(style, date_touple))
        
    return re_time


# �������ʱ���ʽ������
df['apply_time'] = add_time(df.shape[0], 'get_ipython().run_line_magic("Y-%m-%d')", "")
df['job_time'] = add_time(df.shape[0], "get_ipython().run_line_magic("Y/%m/%d", " %H:%M:%S\")")


df['apply_time']


df['job_time']


def add_row(df_temp, num):
    '''
    ��df_temp�����ȡnum������
    '''
    index_1 = np.random.randint(low=0,high=df_temp.shape[0]-1, size = num)
    return df_temp.loc[index_1]


# �����������
df_temp = add_row(df, 10)
df = pd.concat([df, df_temp],axis=0,ignore_index=True)


df.shape


# ����head����������ʾ���л�����ʾȫ��
pd.set_option('display.max_columns', 10)
df.head()


pd.set_option('display.max_columns', None)
df.head()


# ��ɢ�������·�Χ
df.status_account.unique()


# �����ַ���ϴ
df.status_account = df.status_account.apply(lambda x: x.replace(' ','').replace('get_ipython().run_line_magic("','').replace('/t','').replace('$','').replace('@','').replace(';',''))", "")


df.status_account.unique()


# ʱ���ʽͳһ
# ͳһΪ'get_ipython().run_line_magic("Y-%m-%d'", "")
# df['job_time'] = df['job_time'].apply(lambda x:datetime.datetime.strptime( x, 'get_ipython().run_line_magic("d/%m/%Y", " %H:%M:%S.%f))")
df['job_time'] = add_time(df.shape[0], 'get_ipython().run_line_magic("Y-%m-%d')", "")


df['apply_time'] = df['apply_time'].apply(lambda x:datetime.datetime.strptime( x, 'get_ipython().run_line_magic("Y-%m-%d'))", "")


df['apply_time']


# ����ȥ����
df.drop_duplicates(subset=None, keep='first',inplace=True)


df.shape


# ���ն���ȥ����
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


# ȱʧֵ��ͼ
msno.bar(df, labels=True, figsize=(10,6),fontsize=10)


#�����������ݻ�������ͼ���۲��Ƿ����쳣ֵ 
plt.figure(figsize=(10,6)) # ����ͼ�γߴ��С
for j in range(1, len(numerical_var)+1):
    plt.subplot(2,4,j)
    df_temp = df[numerical_var[j-1]][~df[numerical_var[j-1]].isnull()]
    plt.boxplot(df_temp
                ,notch = False # ��λ�ߴ������ð���
                ,widths=0.2    # ����������
                ,medianprops={'color':'red'}  # ��λ������Ϊ��ɫ
                ,boxprops=dict(color="blue")  # ����߿�����Ϊ��ɫ
                ,labels=[numerical_var[j-1]]  # ���ñ�ǩ
                ,whiskerprops={'color':'black'} # ���������ɫ����ɫ
                ,capprops={'color':'green'}     # ��������ͼ���˺�ĩ�˺��ߵ����ԣ���ɫΪ��ɫ
                ,flierprops={'color':'purple','markeredgecolor':'purple'} # �쳣ֵ����
               )
    plt.show()


# ����������ͬ����µķֲ�
for i in numerical_var:
    df_temp = df.loc[~df[i].isnull(),[i,'target']] # ��һ��Ϊ�������ڶ���ΪҪȡ����
    df_good = df_temp[df_temp.target == 0]
    df_bad = df_temp[df_temp.target == 1]
    # ����ͳ����
    valid = round(df_temp.shape[0] / df.shape[0]*100,2)
    Mean = round(df_temp[i].mean(),2)
    Std = round(df_temp[i].std(), 2)
    Max = round(df_temp[i].max(), 2)
    Min = round(df_temp[i].min(), 2)
    
    # ��ͼ
    plt.figure(figsize=(10,6))
    fontsize_1 = 12
    plt.hist(df_good[i], bins=20, alpha=0.5,label='������')
    plt.hist(df_bad[i], bins=20, alpha=0.5, label='������')
    plt.ylabel(i, fontsize=fontsize_1)
    plt.title(f'valid={valid},Mean={Mean},Std={Std},Max={Max},Min={Min}')
    plt.legend()
    file = f"{i}.png"
    plt.savefig(file)
    plt.close(1)


# ��ɢ������ͬ����µķֲ�
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
    plt.bar(np.arange(1,df_2.shape[0]+1),df_2.bin_rate,0.1,color='black',alpha=0.5,label='ռ��')
    plt.xticks(np.arange(1,df_2.shape[0]+1),df_2.var_name)
    plt.plot(np.arange(1,df_2.shape[0]+1),df_2.bad_rate,color='green',alpha=0.5,label='��������')
    plt.ylabel(i,fontsize=fontsize_1)
    plt.title(f'valid rate={valid}get_ipython().run_line_magic("')", "")
    plt.legend()
    file = f"{i}.png"
    plt.savefig(file)
    plt.close(1)



