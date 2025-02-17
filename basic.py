import os
import time
import datetime
import re
import glob
import sqlite3
import datetime
import pandas as pd
import numpy as np
import akshare as ak
from sys import getsizeof as getsize
from operator import methodcaller
from collections import Counter

def init_path(path_dir):
    "创建当前.py目录下的文件夹"
    if os.path.exists(path=path_dir)==bool(False):
        os.mkdir(path=path_dir)
    return None

def get_glob_list(path_dir):
    "返回符合条件的文件名列表"
    # return glob.glob(pathname=path_dir)
    return [os.path.basename(i) for i in glob.iglob(pathname=path_dir,recursive=False)]

def trans_time(ts_list,target_format):
    """
    将ts_list按照任意格式互相转换 target_format: choice of {"string","date","datetime","timestamp"}
    """
    def trans(string):
        "月份/日期转换(01→1)"
        if string[0]!=0:
            result=string
        elif string[0]==0:
            result=string[-1]
        return int(result)

    def str_to_datetime(ts_list):
        formats_8=['%d/%m/%Y','%Y-%m-%d','%Y%m%d','%Y/%m/%d']
        formats_6=["%Y/%m","%Y%m","%Y-%m"]
        formats_4=["%Y"]

        s=ts_list[0]
        s=s.replace("-","")
        s=s.replace("/","")
        if len(s)==8:
            for fmt in formats_8:
                try:
                    L=[datetime.datetime.strptime(i,fmt) for i in ts_list]
                    return L
                except ValueError:
                    pass
            raise ValueError("Invalid date format")
        elif len(s)==6:
            if len(s)==6:
                for fmt in formats_6:
                    try:
                        L=[datetime.datetime.strptime(i, fmt) for i in ts_list]
                        return L
                    except ValueError:
                        pass
                raise ValueError("Invalid date format")
        elif len(s)==4:
            for fmt in formats_4:
                try:
                    L=[datetime.datetime.strptime(i, fmt) for i in ts_list]
                    return L
                except ValueError:
                    pass
            raise ValueError("Invalid date format")

    def trans_from_string(ts_list,target_format):
        # 将字符串转化为其他时间格式
        if target_format=="datetime":
            L=str_to_datetime(ts_list=ts_list)
        elif target_format=="date":
            L=[i.date() for i in str_to_datetime(ts_list=ts_list)]
        elif target_format=="timestamp":
            L=[pd.to_datetime(i) for i in ts_list]
        elif target_format=="string":
            L=ts_list
        return L

    def trans_from_datetime(ts_list,target_format):
        if target_format=="string":
            L=[i.strftime("%Y-%m-%d") for i in ts_list]
        elif target_format=="date":
            L=[i.date() for i in ts_list]
        elif target_format=="timestamp":
            L=[pd.to_datetime(i) for i in ts_list]
        elif target_format=="datetime":
            L=ts_list
        return L

    # step1. target_format为timestamp的直接输出
    if target_format=="timestamp":
        if type(ts_list[0])==type(np.datetime64('2023-12-29T00:00:00.000000000')):   # DolphinDB导出的date格式通常为numpy.datetime64格式
            L=[pd.Timestamp(i) for i in ts_list]
        else:
            L=[pd.to_datetime(i) for i in ts_list]

    # step2. target_format为其他的间接输出
    else:
        """
        int型→string型
        """
        if type(ts_list[0])==type(1):
            ts_list=[str(i) for i in ts_list]
        sample=ts_list[0]
        if type(sample)==type("string"):    # 字符串格式时间{"20200101","2020-01-01"}
            L=trans_from_string(ts_list=ts_list,target_format=target_format)

        elif type(sample)==type(datetime.datetime(2020,1,1,12,0,0)):  # datetime.datetime格式
            L=trans_from_datetime(ts_list=ts_list,target_format=target_format)

        elif type(sample)==type(datetime.date(2020,1,1)):   # datetime.date格式
            datetime_list=[datetime.datetime.strptime(str(i),'%Y-%m-%d') for i in ts_list]  # date→datetime格式→others
            L=trans_from_datetime(ts_list=datetime_list,target_format=target_format)

        elif type(sample)==type(pd.Timestamp('2020-01-01 12:34:56')):   # pandas timestamp格式→string格式→others
            string_list=[i.strftime('%Y-%m-%d %H:%M:%S')[:10] for i in ts_list]
            L=trans_from_string(ts_list=string_list,target_format=target_format)

        else:
            L=[pd.Timestamp(i) for i in ts_list]

    return L

def numeric_df(df,col,errors='ignore'):
    """
    errors: choice of {'ignore','coerce'}
    function+：729,716,576,150 → 729716576150.00
    """
    for i in list(df.columns):
        if i in col:
            try:
                if type(list(df[i])[0])==type("string"):
                    df[i]=[j.replace(",","") for j in list(df[i])]
            except:
                pass
            df[i]=pd.to_numeric(df[i],errors=errors)
            df[i]=df[i].astype(float)
    return df

def concat_df(df_list):
    """
    【新增】：为了解决DataFrame较大情况下内存溢出的现象
    """
    data={}
    for df in df_list:
        for column in df.columns:
            data[column]=df[column].values
    total_df=pd.DataFrame(data,copy=False)
    return total_df

def reverse_Dict(Dict):
    """
    将键值对Dict→值键对Dict
    """
    Dict_new=dict(zip(Dict.values(),Dict.keys()))
    return Dict_new

def to_month_end(ts_list):
    """
    将时间戳转换为月末(仅支持string格式/date格式日期)
    """
    if type(ts_list[0])==type(1):
        ts_list=[str(i) for i in ts_list]
    sample=ts_list[0]
    L=trans_time(ts_list=ts_list,target_format="timestamp")
    month_end_list=[i+pd.offsets.MonthEnd(0) for i in L]
    if type(sample)==type("string"):  # 字符串格式时间{"20200101","2020-01-01"}
        month_end_list=trans_time(month_end_list,"string")
    elif type(sample)==type(datetime.datetime(2020,1,1,12,0,0)):  # datetime.datetime格式
        month_end_list=trans_time(month_end_list,"datetime")
    elif type(sample)==type(datetime.date(2020, 1, 1)):  # datetime.date格式
        month_end_list=trans_time(month_end_list,"date")
    return month_end_list

def element_counter(element_list):
    """
    传入list/list的list，返回对应的element_counter_Dict
    """
    # 判断输入是否为列表或列表的列表
    if isinstance(element_list,list) and all(isinstance(item,list) for item in element_list):
        flattened_list=[i for sublist in element_list for i in sublist]         # 将所有元素展开为一个列表
        counter=Counter(flattened_list) # 使用 Counter 进行统计
        result_Dict=dict(counter)       # 将统计结果转换为字典
        return result_Dict
    elif isinstance(element_list,list):
        counter=Counter(element_list)   # 使用 Counter 进行统计
        result_Dict=dict(counter)       # 将统计结果转换为字典
        return result_Dict


