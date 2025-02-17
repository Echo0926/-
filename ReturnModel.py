import os,sys
import time
import pandas as pd
import numpy as np
import dolphindb as ddb
import tqdm
import matplotlib.pyplot as plt
import streamlit as st
from basic import *
import warnings
warnings.filterwarnings("ignore")
pd.set_option("display.max_columns",None)
plt.rcParams['font.sans-serif']=['KaiTi'] # 显示中文
plt.rcParams['axes.unicode_minus']=False # 显示负号

class ReturnModel_Backtest:
    def __init__(self,session,pool,start_date,end_date,
                 symbol_database,symbol_table,Symbol_prepareFunc,    # 标的数据库(后续市值数据也可以加进去)):
                 factor_database,factor_table,factor_list,Factor_prepareFunc,  # 因子数据库
                 benchmark_database,benchmark_table,benchmark_list,Benchmark_prepareFunc, # 基准收益数据库
                 combine_database,combine_table,Combine_prepareFunc,    # 合并(标的+因子+基准收益)数据库
                 result_database,FactorR_predictFunc,MultiFactorR_predictFunc, # 【新增,COMPO分区的结果数据库】  # WFA根据前t-1期数据预测t期因子收益率
                 posPeriod=5,Multi_Intercept=True,    # 多因子模型是否添加截距项
                 period_return_Algo="nullFill((close-open)/open,0) context by symbol",           # 资产收益率计算公式-1 (每个period内共享该收益率)
                 daily_return_Algo="nullFill((next_price-price)/price-(next_benchmark-benchmark)/benchmark,0)", # 资产收益率计算公式-2 (每个period内的每个日期顺序对应下一个period的日期)
                 ModelR_predictFunc=None,Model_list=None,       # 自定义ML&DL模型相关函数
                 Factor_sliceFunc=None,Asset_sliceFunc=None,Group_list=None,    # 因子&资产分组相关函数
                 optimize_database=None,Optimize_func=None,optstrategy_list=None, # 【新增】:优化函数
                 SingleFactor_estimation=True,MultiFactor_estimation=True,  # 是否单因子估计与多因子估计
                 Ridge_estimation=False,Lasso_estimation=False,ElatsicNet_estimation=False, # 因子估计方法(默认仅OLS,这里可以选择额外的估计方法)
                 Ridge_lamdas=None,Lasso_lamdas=None,ElasticNet_lamdas=None # 5折交叉验证的参数选择
                 ):
        """
        symbol_database: symbol date price industry state
        factor_database: symbol date factor_name factor_value
        benchmark_database: symbol date benchmark_price
        """
        # 基本信息
        self.strategy_name="strategy"
        self.session=session
        self.pool=pool

        # 库表类
        self.symbol_database=symbol_database
        self.symbol_table=symbol_table
        self.factor_database=factor_database
        self.factor_table=factor_table
        self.factor_list=factor_list
        self.benchmark_database=benchmark_database
        self.benchmark_table=benchmark_table
        self.benchmark_list=benchmark_list
        self.combine_database=combine_database  # symbol date
        self.combine_table=combine_table
        self.result_database=result_database    # 【新增:result_database】
        self.optimize_database=optimize_database

        # 函数类
        self.Symbol_prepareFunc=Symbol_prepareFunc
        self.Factor_prepareFunc=Factor_prepareFunc
        self.Benchmark_prepareFunc=Benchmark_prepareFunc
        self.Combine_prepareFunc=Combine_prepareFunc        # 数据合并函数
        self.FactorR_predictFunc=FactorR_predictFunc        # 因子收益率预测函数
        self.MultiFactorR_predictFunc=MultiFactorR_predictFunc  # 因子收益率预测函数(Multi)
        self.ModelIndividualR_predictFunc=ModelR_predictFunc    # 资产收益率预测函数(ML&DL)
        self.Factor_sliceFunc=Factor_sliceFunc  # 因子分组函数
        self.Asset_sliceFunc=Asset_sliceFunc    # 资产分组函数
        self.OptimizeFunc=Optimize_func         # 组合优化函数

        # 变量类
        if isinstance(posPeriod,int) or isinstance(posPeriod,float):
            self.posPeriod=[posPeriod]    # 调仓周期
        self.t=self.posPeriod[0]    # 持仓时间
        self.benchmark=None         # 基准标的
        self.counter=0              # period1...periodN
        self.start_date=start_date
        self.end_date=end_date
        self.start_dot_date=pd.Timestamp(self.start_date).strftime('%Y.%m.%d')
        self.end_dot_date=pd.Timestamp(self.end_date).strftime('%Y.%m.%d')
        self.Period_return_Algo=period_return_Algo        # 资产收益率计算公式-1(每个period内共享该收益率)
        self.Daily_return_Algo=daily_return_Algo          # 资产收益率计算公式-2(每个period的每个交易日顺序对应下一个period的交易日)
        self.SingleFactor_estimation=SingleFactor_estimation    # 是否进行单因子估计
        self.MultiFactor_estimation=MultiFactor_estimation      # 是否进行多因子估计
        self.Multi_Intercept=Multi_Intercept                    # 多因子模型是否添加截距项
        self.Ridge_estimation=Ridge_estimation
        self.Lasso_estimation=Lasso_estimation
        self.ElasticNet_estimation=ElatsicNet_estimation
        if not Ridge_lamdas:
            self.Ridge_lamdas=[0.001,0.01,0.1,1,10,100,1000]
        else:
            self.Ridge_lamdas=Ridge_lamdas
        if not Lasso_lamdas:
            self.Lasso_lamdas=[0.001,0.01,0.1,1,10,100,1000]
        else:
            self.Lasso_lamdas=Lasso_lamdas
        if not ElasticNet_lamdas:
            self.ElasticNet_lamdas=[0.001,0.01,0.1,1,10,100,1000]
        else:
            self.ElasticNet_lamdas=ElasticNet_lamdas
        if not Group_list:
            self.Group_list=["group"]                       # 默认只有一个分组维度
        else:
            self.Group_list=Group_list                      # 分组维度(["Group1",....,"GroupN"]分别代表不同的分组维度)

        # 中间计算+最终结果类
        self.template_table="template"                  # 包含 start_date end_date period,为回测结果的基石
        self.template_daily_table="template_daily"      # 包含 symbol date 两列
        self.template_individual_table="template_individual"    # 包含 symbol start_date end_date period,为回测的所有结果
        self.individualF_table="individual" # 统计每个标的区间因子+下个区间的收益率
        self.individualF_daily_table="individual_daily" # 统计每日的因子+下个区间的收益率

        # 单因子结果
        self.summary_table="summary"        # 向量化回测得到因子统计量
        self.summary_daily_table="summary_daily"
        self.factorR_table="factor_return"   # t-1期数据WFA预测t期因子收益率
        self.factorR_daily_table="factor_return_daily"
        self.individualR_table="individual_return"  # 合并因子收益率,从而得到个股各阶段的real_return(t-t+1时刻真实收益),expect_return(t+1时刻对其拟合),return_pred(根据t-1期数据WFA预测出的因子收益率计算出的个股收益率)

        # 多因子结果
        self.Multisummary_table="Multisummary"  # 多因子向量化回测得到因子统计量
        self.Multisummary_daily_table="Multisummary_daily"
        self.MultifactorR_table="Multifactor_return" # t-1期数据WFA预测t期因子收益率
        self.MultiIndividualR_table="MultiIndividual_return"    # 多因子模型预测的个股收益率

        # 模型预测收益率结果
        self.ModelIndividualR_table="ModelIndividual_return"    # 自定义模型(ML/DL)预测的个股收益率
        if not Model_list:
            self.Model_list=[]
        else:
            self.Model_list=Model_list  # 自定义模型list(顺序不能乱)

        # 资产选择结果
        self.factor_slice_table="factor_slice"      # 因子选择结果表
        self.asset_slice_table="asset_slice"    # 资产选择结果
        self.portfolio_table="portfolio"
        self.signal_table="signal"

        # 组合优化结果
        self.optimize_data_table="optimize_data"   # 投资组合优化信息(用于计算)
        self.optimize_result_table="optimize_result" # 投资组合优化结果
        self.optstrategy_list=optstrategy_list

    def init_SymbolDatabase(self,dropDatabase=False):
        """[Optional]第一次运行,初始化行情数据库"""
        if dropDatabase:
            if session.existsDatabase(dbUrl=self.symbol_database):  # 删除数据库
                self.session.dropDatabase(dbPath=self.symbol_database)
            else:
                pass
        if not self.session.existsTable(dbUrl=self.symbol_database,tableName=self.symbol_table):
            self.session.run(f"""
            db=database("{self.symbol_database}",RANGE,2000.01M+(0..30)*12,engine="TSDB")
            schemaTb=table(1:0,`symbol`date`open`close`marketvalue`state`industry,[SYMBOL,DATE,DOUBLE,DOUBLE,DOUBLE,DOUBLE,SYMBOL]);
            t=db.createPartitionedTable(table=schemaTb,tableName="{self.symbol_table}",partitionColumns="date",sortColumns=["symbol","industry","date"],keepDuplicates=LAST)
            """)
        else:
            pass

    def init_BenchmarkDatabase(self):
        """[Optional]第一次运行,初始化基准收益数据库"""
        if not self.session.existsTable(dbUrl=self.benchmark_database,tableName=self.benchmark_table):
            self.session.run(f"""
            db=database("{self.benchmark_database}",RANGE,2000.01M+(0..30)*12,engine="TSDB")
            schemaTb=table(1:0,`symbol`date`open`close,[SYMBOL,DATE,DOUBLE,DOUBLE]);
            t=db.createPartitionedTable(table=schemaTb,tableName="{self.benchmark_table}",partitionColumns="date",sortColumns=["symbol","date"],keepDuplicates=LAST)
            """)
        else:
            pass

    def init_FactorDatabase(self,dropDatabase=False):
        """[Optional]第一次运行,初始化因子数据库
        【新增】state表示该票当日是否能够交易"""
        if dropDatabase:
            if session.existsDatabase(dbUrl=self.factor_database):  # 删除数据库
                self.session.dropDatabase(dbPath=self.factor_database)
            else:
                pass
        if not self.session.existsTable(dbUrl=self.factor_database,tableName=self.factor_table):
            self.session.run(f"""
            // 创建因子数据库
            create database "{self.factor_database}" 
            partitioned by RANGE(date(datetimeAdd(2000.01M,0..30*12,"M"))), VALUE(`f1`f2),   // 默认两个因子分区,后面再加
            engine='TSDB'
            
            create table "{self.factor_database}"."{self.factor_table}"(
                symbol SYMBOL, 
                date DATE[comment="时间列", compress="delta"], 
                factor_name SYMBOL, 
                factor_value DOUBLE
            )
            partitioned by date, factor_name,
            sortColumns=[`symbol,`date], 
            keepDuplicates=LAST, 
            sortKeyMappingFunction=[hashBucket{{, 500}}];
            
            // 添加数据库分区
            for (factor in {self.factor_list}){{
                addValuePartitions(database("{self.factor_database}"),string(factor),1);  // DolphinDB会自动判断是否存在现有数据分区
            }};
            """)
        else:
            pass

    def init_CombineDataBase(self):
        """[Necessary]初始化合并数据库+模板数据库"""
        # Combine Table 默认每次回测前删除上次的因子库（因为因子个数名称&调仓周期可能不一样）
        if self.session.existsTable(dbUrl=self.combine_database,tableName=self.combine_table):
            self.session.dropTable(dbPath=self.combine_database,tableName=self.combine_table)
        L=[]
        columns_name=["symbol","date","open","close","marketvalue","state","industry"]+["period"]+[f"{i}_open" for i in self.benchmark_list]+[f"{i}_close" for i in self.benchmark_list]+self.factor_list
        columns_type=["SYMBOL","DATE","DOUBLE","DOUBLE","DOUBLE","DOUBLE","SYMBOL"]+["DOUBLE"]+["DOUBLE"]*len(self.benchmark_list)+["DOUBLE"]*len(self.benchmark_list)+["DOUBLE"]*len(self.factor_list)
        self.session.run(f"""
        db=database("{self.combine_database}",RANGE,2000.01M+(0..30)*12,engine="TSDB");
        schemaTb=table(1:0,{columns_name},{columns_type});
        t=db.createPartitionedTable(table=schemaTb,tableName="{self.combine_table}",partitionColumns="date",sortColumns=["symbol","date"],keepDuplicates=LAST)
        """)

        # Template Table 默认每次回测时候删除上次的模板库
        if self.session.existsTable(dbUrl=self.combine_database,tableName=self.template_table):
            self.session.dropTable(dbPath=self.combine_database,tableName=self.template_table)
        columns_name=["start_date","end_date","period"]
        columns_type=["DATE","DATE","DOUBLE"]
        self.session.run(f"""
        db=database("{self.combine_database}");
        schemaTb=table(1:0,{columns_name},{columns_type});
        t=db.createPartitionedTable(table=schemaTb,tableName="{self.template_table}",partitionColumns="start_date",sortColumns=["start_date"])
        """)

        # Template Individual Table 默认每次回测时候删除上次的模板库
        if self.session.existsTable(dbUrl=self.combine_database,tableName=self.template_individual_table):
            self.session.dropTable(dbPath=self.combine_database,tableName=self.template_individual_table)
        columns_name=["symbol","start_date","end_date","period"]
        columns_type=["SYMBOL","DATE","DATE","DOUBLE"]
        self.session.run(f"""
        db=database("{self.combine_database}");
        schemaTb=table(1:0,{columns_name},{columns_type});
        t=db.createPartitionedTable(table=schemaTb,tableName="{self.template_individual_table}",partitionColumns="start_date",sortColumns=["symbol","start_date"],keepDuplicates=LAST)
        """)

        # Template Daily Table 默认每次回测时候删除上次的模板库
        if self.session.existsTable(dbUrl=self.combine_database,tableName=self.template_daily_table):
            self.session.dropTable(dbPath=self.combine_database,tableName=self.template_daily_table)
        columns_name=["symbol","date","period"]
        columns_type=["SYMBOL","DATE","DOUBLE"]
        self.session.run(f"""
        db=database("{self.combine_database}");
        schemaTb=table(1:0,{columns_name},{columns_type});
        t=db.createPartitionedTable(table=schemaTb,tableName="{self.template_daily_table}",partitionColumns="date",sortColumns=["symbol","date"],keepDuplicates=LAST)
        """)

    def init_ResultDataBase(self,dropDatabase=False):
        """单因子&多因子结果&Structured Data数据库"""
        if dropDatabase:
            if session.existsDatabase(dbUrl=self.result_database):  # 删除数据库
                self.session.dropDatabase(dbPath=self.result_database)
            else:
                pass

        # individual_factor(current_factor_value+current_period_return)
        if self.session.existsTable(dbUrl=self.result_database,tableName=self.individualF_table):
            self.session.dropTable(dbPath=self.result_database,tableName=self.individualF_table)
        columns_name=["Benchmark","period","symbol"]+self.factor_list+["real_return"]
        columns_type=["SYMBOL","DOUBLE","SYMBOL"]+["DOUBLE"]*len(self.factor_list)+["DOUBLE"]
        self.session.run(f"""
        db=database("{self.result_database}",LIST,{self.benchmark_list},engine="OLAP");
        schemaTb=table(1:0,{columns_name},{columns_type});
        t=db.createPartitionedTable(table=schemaTb,tableName="{self.individualF_table}",partitionColumns=["Benchmark"])
        """)

        # individual_factor(daily_factor_value+next_period_return)
        if self.session.existsTable(dbUrl=self.result_database,tableName=self.individualF_daily_table):
            self.session.dropTable(dbPath=self.result_database,tableName=self.individualF_daily_table)
        columns_name=["Benchmark","date","symbol"]+self.factor_list+["real_return"]
        columns_type=["SYMBOL","DATE","SYMBOL"]+["DOUBLE"]*len(self.factor_list)+["DOUBLE"]
        self.session.run(f"""
        db=database("{self.result_database}");
        schemaTb=table(1:0,{columns_name},{columns_type});
        t=db.createPartitionedTable(table=schemaTb,tableName="{self.individualF_daily_table}",partitionColumns=["Benchmark"])
        """)

        if self.SingleFactor_estimation:    # 如果需要进行单因子测试
            # 单因子模型数据库
            # summary_result(model eval+Factor Return+Factor IC+Factor t)
            if self.session.existsTable(dbUrl=self.result_database,tableName=self.summary_table):
                self.session.dropTable(dbPath=self.result_database,tableName=self.summary_table)
            columns_name=["Benchmark","period","class","indicator","value"]
            columns_type=["SYMBOL","DOUBLE","SYMBOL","SYMBOL","DOUBLE"]
            self.session.run(f"""
            db=database("{self.result_database}");
            schemaTb=table(1:0,{columns_name},{columns_type});
            t=db.createPartitionedTable(table=schemaTb,tableName="{self.summary_table}",partitionColumns=["Benchmark"])
            """)

            # summary_result(daily)(model eval+Factor Return+Factor IC+Factor t)
            if self.session.existsTable(dbUrl=self.result_database,tableName=self.summary_daily_table):
                self.session.dropTable(dbPath=self.result_database,tableName=self.summary_daily_table)
            columns_name=["Benchmark","date","class","indicator","value"]
            columns_type=["SYMBOL","DATE","SYMBOL","SYMBOL","DOUBLE"]
            self.session.run(f"""
            db=database("{self.result_database}");
            schemaTb=table(1:0,{columns_name},{columns_type});
            t=db.createPartitionedTable(table=schemaTb,tableName="{self.summary_daily_table}",partitionColumns=["Benchmark"])
            """)

            # factorR_result
            if self.session.existsTable(dbUrl=self.result_database,tableName=self.factorR_table):
                self.session.dropTable(dbPath=self.result_database,tableName=self.factorR_table)
            columns_name=["Benchmark","period","class","indicator","value","value_pred"]
            columns_type=["SYMBOL","DOUBLE","SYMBOL","SYMBOL","DOUBLE","DOUBLE"]
            self.session.run(f"""
            db=database("{self.result_database}");
            schemaTb=table(1:0,{columns_name},{columns_type});
            t=db.createPartitionedTable(table=schemaTb,tableName="{self.factorR_table}",partitionColumns=["Benchmark"])
            """)

            # individualR_result
            if self.session.existsTable(dbUrl=self.result_database,tableName=self.individualR_table):
                self.session.dropTable(dbPath=self.result_database,tableName=self.individualR_table)
            columns_name=["Benchmark","period","symbol","real_return","method"]
            columns_type=["SYMBOL","DOUBLE","SYMBOL","DOUBLE","SYMBOL"]
            for col in self.factor_list:
                columns_name=columns_name+[str(col)+"_return_pred"]
                columns_type=columns_type+["DOUBLE"]
            self.session.run(f"""
            db=database("{self.result_database}");
            schemaTb=table(1:0,{columns_name},{columns_type});
            t=db.createPartitionedTable(table=schemaTb,tableName="{self.individualR_table}",partitionColumns=["Benchmark"])
            """)
        else:
            pass

        if self.MultiFactor_estimation: # 如果需要进行多因子测试
            # 多因子模型数据库
            # Multisummary_result(model eval+Factor Return+Factor IC+Factor t)
            if self.session.existsTable(dbUrl=self.result_database,tableName=self.Multisummary_table):
                self.session.dropTable(dbPath=self.result_database,tableName=self.Multisummary_table)
            columns_name=["Benchmark","period","class","indicator","value"]
            columns_type=["SYMBOL","DOUBLE","SYMBOL","SYMBOL","DOUBLE"]
            self.session.run(f"""
            db=database("{self.result_database}");
            schemaTb=table(1:0,{columns_name},{columns_type});
            t=db.createPartitionedTable(table=schemaTb,tableName="{self.Multisummary_table}",partitionColumns=["Benchmark"])
            """)

            # MultifactorR_result
            if self.session.existsTable(dbUrl=self.result_database,tableName=self.MultifactorR_table):
                self.session.dropTable(dbPath=self.result_database,tableName=self.MultifactorR_table)
            columns_name=["Benchmark","period","class","indicator","value","value_pred"]
            columns_type=["SYMBOL","DOUBLE","SYMBOL","SYMBOL","DOUBLE","DOUBLE"]
            self.session.run(f"""
            db=database("{self.result_database}");
            schemaTb=table(1:0,{columns_name},{columns_type});
            t=db.createPartitionedTable(table=schemaTb,tableName="{self.MultifactorR_table}",partitionColumns=["Benchmark"])
            """)

            # MultiIndividualR_result(相比IndividualR_result少了一列indicator,因为是所有因子合在一起预测的结果)
            if self.session.existsTable(dbUrl=self.result_database,tableName=self.MultiIndividualR_table):
                self.session.dropTable(dbPath=self.result_database,tableName=self.MultiIndividualR_table)
            L=[]
            for i,j in zip([self.Lasso_estimation,self.Ridge_estimation,self.ElasticNet_estimation],
                            ["Lasso","Ridge","ElasticNet"]):
                if i:
                    L.append(j)
            columns_name=["Benchmark","period","symbol","real_return"]+["return_pred_OLS"]+[f"return_pred_{i}" for i in L]
            columns_type=["SYMBOL","DOUBLE","SYMBOL","DOUBLE"]+["DOUBLE"]+["DOUBLE"]*len(L)
            self.session.run(f"""
            db=database("{self.result_database}");
            schemaTb=table(1:0,{columns_name},{columns_type});
            t=db.createPartitionedTable(table=schemaTb,tableName="{self.MultiIndividualR_table}",partitionColumns=["Benchmark"])
            """)
        else:
            pass

    def init_ModelDatabase(self):
        """自定义模型(ML/DL)资产收益率预测数据库"""
        # 默认删除原来的数据库
        if self.session.existsTable(dbUrl=self.result_database,tableName=self.ModelIndividualR_table):
            self.session.dropTable(dbPath=self.result_database,tableName=self.ModelIndividualR_table)
        if not self.session.existsTable(dbUrl=self.result_database,tableName=self.ModelIndividualR_table):
            columns_name=["Benchmark","period","symbol","real_return"]+[str(i)+"_return_pred" for i in self.Model_list]
            columns_type=["SYMBOL","DOUBLE","SYMBOL","DOUBLE"]+["DOUBLE"]*len(self.Model_list)
            self.session.run(f"""
            db=database("{self.result_database}");
            schemaTb=table(1:0,{columns_name},{columns_type});
            t=db.createPartitionedTable(table=schemaTb,tableName="{self.ModelIndividualR_table}",partitionColumns=["Benchmark"])
            """)

    def init_SliceDatabase(self):
        """因子选择&资产选择数据库"""
        # factor_slice 因子选择(不适用于多因子模型)
        if self.session.existsTable(dbUrl=self.result_database,tableName=self.factor_slice_table):
            self.session.dropTable(dbPath=self.result_database,tableName=self.factor_slice_table)
        columns_name=["Benchmark","period","indicator","target"]
        columns_type=["SYMBOL","DOUBLE","SYMBOL","DOUBLE"]
        self.session.run(f"""
        db=database("{self.result_database}");
        schemaTb=table(1:0,{columns_name},{columns_type});
        t=db.createPartitionedTable(table=schemaTb,tableName="{self.factor_slice_table}",partitionColumns=["Benchmark"])
        """)

        # asset_slice 资产标的选择
        # target:即Group
        if self.session.existsTable(dbUrl=self.result_database,tableName=self.asset_slice_table):
            self.session.dropTable(dbPath=self.result_database,tableName=self.asset_slice_table)
        columns_name=["Benchmark","period","symbol"]+self.Group_list
        columns_type=["SYMBOL","DOUBLE","SYMBOL"]+len(self.Group_list)*["DOUBLE"]
        self.session.run(f"""
        db=database("{self.result_database}");
        schemaTb=table(1:0,{columns_name},{columns_type});
        t=db.createPartitionedTable(table=schemaTb,tableName="{self.asset_slice_table}",partitionColumns=["Benchmark"])
        """)

    def init_OptimizeDatabase(self,dropDatabase=False):
        """【新增】initOptimizeDatabase(COMPO Database)"""
        if dropDatabase:
            if session.existsDatabase(dbUrl=self.optimize_database):  # 删除数据库
                self.session.dropDatabase(dbPath=self.optimize_database)
            else:
                pass
        # optimize_data(用于组合优化的数据库)
        if self.session.existsTable(dbUrl=self.optimize_database,tableName=self.optimize_data_table):
            self.session.dropTable(dbPath=self.optimize_database,tableName=self.optimize_data_table)
        if self.SingleFactor_estimation and self.MultiFactor_estimation:    # 既有单因子结果又有多因子结果
            columns_name=["Benchmark","period","symbol","real_return","method"]+[f"{i}_return_pred" for i in self.factor_list]+["Multi_return_pred"]+[f"{i}_return_pred" for i in self.Model_list]+["marketvalue","industry"]+self.Group_list
            columns_type=["SYMBOL","DOUBLE","SYMBOL","DOUBLE","SYMBOL"]+["DOUBLE"]*len(self.factor_list)+["DOUBLE"]+["DOUBLE"]*len(self.Model_list)+["DOUBLE","SYMBOL"]+["DOUBLE"]*len(self.Group_list)
        elif self.SingleFactor_estimation and not self.MultiFactor_estimation:  # 只有单因子结果
            columns_name=["Benchmark","period","symbol","real_return","method"]+[f"{i}_return_pred" for i in self.factor_list]+[f"{i}_return_pred" for i in self.Model_list]+["marketvalue","industry"]+self.Group_list
            columns_type=["SYMBOL","DOUBLE","SYMBOL","DOUBLE","SYMBOL"]+["DOUBLE"]*len(self.factor_list)+["DOUBLE"]*len(self.Model_list)+["DOUBLE","SYMBOL"]+["DOUBLE"]*len(self.Group_list)
        elif self.MultiFactor_estimation and not self.SingleFactor_estimation:  # 只有多因子结果
            columns_name=["Benchmark","period","symbol","real_return","method"]+["Multi_return_pred"]+[f"{i}_return_pred" for i in self.Model_list]+["marketvalue","industry"]+self.Group_list
            columns_type=["SYMBOL","DOUBLE","SYMBOL","DOUBLE","SYMBOL"]+["DOUBLE"]+["DOUBLE"]*len(self.Model_list)+["DOUBLE","SYMBOL"]+["DOUBLE"]*len(self.Group_list)
        self.session.run(f"""
        db=database("{self.optimize_database}",LIST,{self.benchmark_list},engine="OLAP");
        schemaTb=table(1:0,{columns_name},{columns_type});
        t=db.createPartitionedTable(table=schemaTb,tableName="{self.optimize_data_table}",partitionColumns=["Benchmark"])
        """)

        # optimize_result(组合优化结果数据库)
        if self.session.existsTable(dbUrl=self.optimize_database,tableName=self.optimize_result_table):
            self.session.dropTable(dbPath=self.optimize_database,tableName=self.optimize_result_table)
        if not session.existsTable(dbUrl=self.optimize_database,tableName=self.optimize_result_table):
            columns_name=["Benchmark","period","symbol"]+self.optstrategy_list  # 优化投资组合
            columns_type=["SYMBOL","DOUBLE","SYMBOL"]+["DOUBLE"]*len(self.optstrategy_list)
            self.session.run(f"""
            db=database("{self.optimize_database}");
            schemaTb=table(1:0,{columns_name},{columns_type});
            t=db.createPartitionedTable(table=schemaTb,tableName="{self.optimize_result_table}",partitionColumns=["Benchmark"])
            """)

    def add_SymbolData(self):
        self.Symbol_prepareFunc(self)

    def add_FactorData(self):
        self.Factor_prepareFunc(self)

    def add_BenchmarkData(self):
        self.Benchmark_prepareFunc(self)

    def add_CombineData(self):
        self.Combine_prepareFunc(self)

    def pred_FactorR(self):
        """[单因子]利用t-1期因子收益率预测t期因子收益率函数"""
        self.FactorR_predictFunc(self)

    def pred_MultiFactorR(self):
        """[多因子]利用t-1期因子收益率预测t期因子收益率的函数"""
        self.MultiFactorR_predictFunc(self)

    def pred_ModelIndividualR(self,**params):
        """[多因子]利用t-1期因子值训练模型并预测t期资产收益率的函数"""
        self.ModelIndividualR_predictFunc(self,**params)

    def slice_Factor(self):
        """因子筛选函数"""
        self.Factor_sliceFunc(self)

    def slice_Asset(self):
        """资产标的筛选函数"""
        self.Asset_sliceFunc(self)

    def summary_command(self):
        """[单因子回测]individual_return(period_return)&summary_result&summary_daily_result"""
        return rf"""
        // 单因子回测框架
        pt=select * from loadTable("{self.combine_database}","{self.combine_table}");
        sortBy!(pt,[`symbol,`date],[1,1]);
        factor_list={self.factor_list}; // 因子列表
        Ridge_estimation=int({int(self.Ridge_estimation)});  // 是否估计Ridge因子收益率
        Lasso_estimation=int({int(self.Lasso_estimation)});  // 是否估计Lasso因子收益率
        ElasticNet_estimation=int({int(self.ElasticNet_estimation)});   // 是否估计ElasticNet因子收益率
        
        // 在不同的benchmark+period下计算预期收益率
        for (benchmark_str in ["{self.benchmark}"]){{    
            //【0.Preparation】 
            pt[`benchmark_open]=pt[benchmark_str+"_open"];   // 当前benchmark对应的开盘价
            pt[`benchmark_close]=pt[benchmark_str+"_close"]; // 当前benchmark对应的收盘价
            
            //【1.PosOLS】 【挣扎的点:时间点的选择→firstNot or lastNot,其实这个取决于你的因子库是长什么样子的,但还是lastNot好一点其实,因为大部分因子是很难在第一天早上就得出的】
            pt[`period_symbol]=string(pt[`period])+pt[`symbol];
            update pt set R={self.Period_return_Algo};  // [自定义]资产收益率计算公式
            pos_pt=select firstNot(symbol) as symbol,firstNot(period) as period, firstNot(R) as R,{','.join(f"first({item}) as {item}" for item in self.factor_list)} from pt group by period_symbol;
            sortBy!(pos_pt,`symbol`period,[1,1]); // 按升序排序
                
            // 1.1.Individual结果 
            // [新增]individual_return保存
            individual_return=pos_pt.copy();    // 复制一份,以免和下面summary结果计算部分冲突
            individual_return=select benchmark_str as Benchmark,period,symbol,{','.join(self.factor_list)},R from individual_return;
            loadTable('{self.result_database}','{self.individualF_table}').append!(individual_return);
            undef(`individual_return);
            
            if (int({int(self.SingleFactor_estimation)})==1){{  // 说明需要进行单因子测试     
                // 1.2.Summary结果
                COUNTER=0;
                distinct_period_list=sort(exec distinct(period) from pos_pt,true);
                for (p in distinct_period_list){{  
                    // Data
                    reg_df=select * from pos_pt where period=p and not isNull(R); // 【新增】去除了收益率空缺值的样本进行回归
                    func=def(X):countNanInf(X,true);
                    reg_df[`naninf_count]=byRow(func,reg_df[factor_list]);
                    reg_df=select * from reg_df where naninf_count=0;
                    
                    if (count(reg_df)>0){{
                        // IC&RankIC
                        IC=[];
                        RankIC=[];
                        for (col in factor_list){{
                            append!(IC,corr(reg_df[col],reg_df[`R]));
                            append!(RankIC,spearmanr(reg_df[col],reg_df[`R]));
                        }};
                        IC_df=table(factor_list as `indicator,IC as `value);
                        IC_df=select `IC as class, indicator, value from IC_df;
                        RankIC_df=table(factor_list as `indicator,RankIC as `value);
                        RankIC_df=select `RankIC as class, indicator, value from RankIC_df;
                        
                        //OLS
                        counter=0;
                        for (col in factor_list){{
                            reg_df[col+"_alpha"]=1.0; // 添加alpha
                            result_OLS=ols(reg_df[`R],reg_df[[col+"_alpha",col]],intercept=false,mode=2);  // OLS回归结果
                                
                            // 统计结果(summary_result,OLS)
                            beta_df=select "R_OLS" as class,factor as indicator,beta as value from result_OLS[`Coefficient];
                            tstat_df=select "tstat_OLS" as class,factor as indicator,tstat as value from result_OLS[`Coefficient];
                            RegDict=dict(result_OLS[`RegressionStat][`item],result_OLS[`RegressionStat][`statistics]);
                            R_square=RegDict[`R2];
                            Adj_square=RegDict[`AdjustedR2];
                            Std_error=RegDict[`StdError];
                            Obs=RegDict['Observations'];
                            
                            // 添加至summary_table的数据行
                            if (counter==0){{
                                summary_result=table([`R_square_OLS,`Adj_square_OLS,`Std_Error_OLS,`Obs_OLS] as `class, [col,col,col,col] as `indicator,[R_square,Adj_square,Std_error,Obs] as `value);}};
                            else{{
                                summary_result.append!(table([`R_square_OLS,`Adj_square_OLS,`Std_Error_OLS,`Obs_OLS] as `class, [col,col,col,col] as `indicator,[R_square,Adj_square,Std_error,Obs] as `value))}};
                            summary_result.append!(beta_df);
                            summary_result.append!(tstat_df);
                                
                            // 统计结果(summary_result,Lasso)
                            if (Lasso_estimation==1){{
                                result_Lasso=lassoCV(reg_df,`R,[col+"_alpha",col],alphas={self.Lasso_lamdas},intercept=false); // Lasso回归结果
                                beta_df=table(result_Lasso[`xColNames] as indicator,result_Lasso[`coefficients] as value);
                                beta_df=select "R_Lasso" as class,* from beta_df;
                                summary_result.append!(beta_df);
                            }};
                            
                            // 统计结果(summary_result,Ridge)
                            if (Ridge_estimation==1){{
                                result_Ridge=ridgeCV(reg_df,`R,[col+"_alpha",col],alphas={self.Ridge_lamdas},intercept=false); // Ridge回归结果
                                beta_df=table(result_Ridge[`xColNames] as indicator,result_Ridge[`coefficients] as value);
                                beta_df=select "R_Ridge" as class,* from beta_df;
                                summary_result.append!(beta_df);
                            }};
                            
                            // 统计结果(summary_result,ElasticNet)
                            if (ElasticNet_estimation==1){{
                                result_ElasticNet=elasticNetCV(reg_df,`R,[col+"_alpha",col],alphas={self.ElasticNet_lamdas},intercept=false); // Ridge回归结果
                                beta_df=table(result_ElasticNet[`xColNames] as indicator,result_ElasticNet[`coefficients] as value);
                                beta_df=select "R_ElasticNet" as class,* from beta_df;
                                summary_result.append!(beta_df);
                            }};
                            counter=counter+1;
                        }};
                        summary_result.append!(IC_df);
                        summary_result.append!(RankIC_df);
                        summary_result=select p as period,* from summary_result;  // 最后添加日期
                            
                        if (COUNTER==0){{
                            final_pos_result=summary_result.copy();}};
                        else{{
                            final_pos_result.append!(summary_result);}};
                        COUNTER=COUNTER+1;
                    }}; // 当reg_df的元素存在时END
                    else{{}}; 
                }};
                undef(`pos_pt`reg_df); // 内存释放
                    
                // 格式调整!!
                final_pos_result=select benchmark_str as Benchmark,period,class,indicator,value from final_pos_result;
                loadTable('{self.result_database}','{self.summary_table}').append!(final_pos_result);
                undef(`final_pos_result); // 内存释放
            }};  //单因子测试部分END
            
            // 【2.DailyOLS】
            pt[`period_symbol]=string(pt[`symbol])+"period"+string(pt[`period]);
            update pt set dateidx=cumcount(date) context by period_symbol;  // dateidx表示每个资产每个周期内date的标号,根据标号left join price与next_price，从而得到日频的period_return
            next_pt=select period-1 as period,symbol,dateidx,open as next_open,close as next_close,benchmark_open as next_benchmark_open,benchmark_close as next_benchmark_close from pt;  // 表示下一期period的price(next_price)
            pt=select * from pt left join next_pt on pt.period=next_pt.period and pt.dateidx=next_pt.dateidx and pt.symbol=next_pt.symbol; 
            undef(`next_pt);
            update pt set open=nullFill(open,close);
            update pt set next_open=nullFill(next_open,next_close);
            update pt set benchmark_open=nullFill(benchmark_open,benchmark_close);
            update pt set next_benchmark_open=nullFill(next_benchmark_open,next_benchmark_close);
            update pt set DailyR={self.Daily_return_Algo};
            dropColumns!(pt,`dateidx`open`next_open`close`next_close`benchmark_open`next_benchmark_open`benchmark_close`next_benchmark_close`period_symbol);
            daily_pt=select first(DailyR) as R,{','.join(f"firstNot({item}) as {item}" for item in self.factor_list)} from pt group by symbol,date;
            sortBy!(daily_pt,`symbol`date,[1,1]); // 按升序排序
                
            // 2.1.Individual结果【注：只有需要ML/DL方法的时候才需要保存这个数据库】 
            // [新增]individual_daily_return保存
            individual_return=select benchmark_str as Benchmark,date,symbol,{','.join(self.factor_list)},R as period_return from daily_pt;
            loadTable('{self.result_database}','{self.individualF_daily_table}').append!(individual_return);
            undef(`individual_return);
            
            if (int({int(self.SingleFactor_estimation)})==1){{ // 说明需要进行单因子测试    
                // 结果列
                COUNTER=0;
                distinct_date_list=sort(exec distinct(date) from daily_pt,true);
                for (t in distinct_date_list){{  // 最后一个date肯定没有下一个date的数据
                    // Data
                    reg_df=select * from daily_pt where date=t and not isNull(R); // 【新增】去除了收益率为空的样本
                    func=def(X):countNanInf(X,true);
                    reg_df[`naninf_count]=byRow(func,reg_df[factor_list]);
                    reg_df=select * from reg_df where naninf_count=0;
                    
                    if (count(reg_df)>0){{  // 因为怕有的date没有数据
                        // IC&RankIC
                        IC=[];
                        RankIC=[];
                        for (col in factor_list){{
                            append!(IC,corr(reg_df[col],reg_df[`R]));
                            append!(RankIC,spearmanr(reg_df[col],reg_df[`R]));}};
                        IC_df=table(factor_list as `indicator,IC as `value);
                        IC_df=select `IC as class, indicator, value from IC_df;
                        RankIC_df=table(factor_list as `indicator,RankIC as `value);
                        RankIC_df=select `RankIC as class, indicator, value from RankIC_df;
                    
                        //OLS
                        counter=0;
                        for (col in factor_list){{
                            reg_df[col+"_alpha"]=1; // 添加alpha
                            result_df=ols(reg_df[`R],reg_df[[col+"_alpha",col]],false,2);
    
                            // 统计结果(summary_result)
                            beta_df=select "R" as class,factor as indicator,beta as value from result_df[`Coefficient];
                            tstat_df=select "tstat" as class,factor as indicator,tstat as value from result_df[`Coefficient];
                            RegDict=dict(result_df[`RegressionStat][`item],result_df[`RegressionStat][`statistics]);
                            R_square=RegDict[`R2];
                            Adj_square=RegDict[`AdjustedR2];
                            Std_error=RegDict[`StdError];
                            Obs=RegDict['Observations'];
                    
                            // 添加至summary_table的数据行
                            if (counter==0){{
                                summary_result=table([`R_square,`Adj_square,`Std_Error,`Obs] as `class, [col,col,col,col] as `indicator,[R_square,Adj_square,Std_error,Obs] as `value);}};
                            else{{
                                summary_result.append!(table([`R_square,`Adj_square,`Std_Error,`Obs] as `class, [col,col,col,col] as `indicator,[R_square,Adj_square,Std_error,Obs] as `value))}};
                            summary_result.append!(beta_df);
                            summary_result.append!(tstat_df);
                            counter=counter+1;
                        }};
                        summary_result.append!(IC_df);
                        summary_result.append!(RankIC_df);
                        summary_result=select t as date,* from summary_result;  // 最后添加日期
                            
                        if (COUNTER==0){{
                            final_daily_result=summary_result.copy();}};
                        else{{
                            final_daily_result.append!(summary_result)}};
                        COUNTER=COUNTER+1;
                    }};
                    else{{}};
                }};
                undef(`daily_pt`reg_df); // 内存释放
                final_daily_result=select benchmark_str as Benchmark,* from final_daily_result;
                loadTable('{self.result_database}','{self.summary_daily_table}').append!(final_daily_result);
                undef(`final_daily_result); // 内存释放
            }}; // 单因子测试END
        
        }}; // period循环END
        undef(`pt`template_pt`template_daily_pt); //释放内存
        """

    def Multisummary_command(self):
        """[多因子回测]"""
        return rf"""
        // 多因子回测框架
        pt=select * from loadTable("{self.combine_database}","{self.combine_table}");
        sortBy!(pt,[`symbol,`date],[1,1]);
        factor_list={self.factor_list}; // 多因子列表
        Ridge_estimation=int({int(self.Ridge_estimation)});  // 是否估计Ridge因子收益率
        Lasso_estimation=int({int(self.Lasso_estimation)});  // 是否估计Lasso因子收益率
        ElasticNet_estimation=int({int(self.ElasticNet_estimation)});   // 是否估计ElasticNet因子收益率
        add_Intercept=int({int(self.Multi_Intercept)});      // 多因子模型是否添加截距项
        
        // 在不同的benchmark+period下计算预期收益率
        for (benchmark_str in ["{self.benchmark}"]){{    
            //【0.Preparation】
            pt[`benchmark_open]=pt[benchmark_str+"_open"];   // 当前benchmark对应的开盘价
            pt[`benchmark_close]=pt[benchmark_str+"_close"]; // 当前benchmark对应的收盘价
            
            //【1.PosOLS】 【挣扎的点:时间点的选择→firstNot or lastNot,其实这个取决于你的因子库是长什么样子的,但还是lastNot好一点其实,因为大部分因子是很难在第一天早上就得出的】
            pt[`period_symbol]=string(pt[`period])+pt[`symbol];
            update pt set R={self.Period_return_Algo};  // [自定义]资产收益率计算公式
            pos_pt=select firstNot(symbol) as symbol,firstNot(period) as period, firstNot(R) as R,{','.join(f"first({item}) as {item}" for item in self.factor_list)} from pt group by period_symbol;
            sortBy!(pos_pt,`symbol`period,[1,1]); // 按升序排序
                 
            // 1.1.MultiSummary结果
            counter=0;
            distinct_period_list=sort(exec distinct(period) from pos_pt,true);
            for (p in distinct_period_list){{  // 最后一个period肯定没有下一个period的数据
                // Data
                reg_df=select * from pos_pt where period=p and not isNull(R); // 【新增】去除了收益率为空的样本
                func=def(X):countNanInf(X,true);
                reg_df[`naninf_count]=byRow(func,reg_df[factor_list]);
                reg_df=select * from reg_df where naninf_count=0;
                
                if (count(reg_df)>0){{
                    // IC&RankIC
                    IC=[];
                    RankIC=[];
                    for (col in factor_list){{
                        append!(IC,corr(reg_df[col],reg_df[`R]));
                        append!(RankIC,spearmanr(reg_df[col],reg_df[`R]));
                    }};
                    IC_df=table(factor_list as `indicator,IC as `value);
                    IC_df=select `IC as class, indicator, value from IC_df;
                    RankIC_df=table(factor_list as `indicator,RankIC as `value);
                    RankIC_df=select `RankIC as class, indicator, value from RankIC_df;
                    
                    //OLS(多因子回测)
                    result_OLS=ols(reg_df[`R],reg_df[factor_list],intercept=true,mode=2);  // 这里添加截距项,所以多因子模型factor_list不能有截距项
                        
                    // 统计结果(Multisummary_result,OLS)
                    beta_df=select "R_OLS" as class,factor as indicator,beta as value from result_OLS[`Coefficient];
                    tstat_df=select "tstat_OLS" as class,factor as indicator,tstat as value from result_OLS[`Coefficient];
                    RegDict=dict(result_OLS[`RegressionStat][`item],result_OLS[`RegressionStat][`statistics]);
                    R_square=RegDict[`R2];
                    Adj_square=RegDict[`AdjustedR2];
                    Std_error=RegDict[`StdError];
                    Obs=RegDict['Observations'];
                        
                    // 添加至summary_table的数据行
                    summary_result=table([`R_square_OLS,`Adj_square_OLS,`Std_Error_OLS,`Obs_OLS] as `class,[`R_square,`Adj_square,`Std_Error,`Obs]  as `indicator,[R_square,Adj_square,Std_error,Obs] as `value);
                    summary_result.append!(beta_df);
                    summary_result.append!(tstat_df);
                        
                    // 统计结果(Multisummary_result,Lasso)
                    if (Lasso_estimation==1){{
                         if (add_Intercept==1){{  // 添加截距项
                            result_Lasso=lassoCV(reg_df,`R,factor_list,alphas={self.Lasso_lamdas},intercept=true);
                            result_Lasso=dict([`intercept].append!(result_Lasso[`xColNames]),[result_Lasso[`intercept]].append!(result_Lasso[`coefficients]));
                    }};else{{ // 不添加截距项
                            result_Lasso=lassoCV(reg_df,`R,factor_list,alphas={self.Lasso_lamdas},intercept=false);
                            result_Lasso=dict(result_Lasso[`xColNames],result_Lasso[`coefficients]);
                    }};
                    beta_df=table(result_Lasso.keys() as indicator,result_Lasso.values() as value);
                    beta_df=select "R_Lasso" as class,* from beta_df;
                    summary_result.append!(beta_df);
                    }};
                        
                    // 统计结果(Multisummary_result,Ridge)
                    if (Ridge_estimation==1){{
                        if (add_Intercept==1){{  // 添加截距项
                            result_Ridge=ridgeCV(reg_df,`R,factor_list,alphas={self.Ridge_lamdas},intercept=true);
                            result_Ridge=dict([`intercept].append!(result_Ridge[`xColNames]),[result_Ridge[`intercept]].append!(result_Ridge[`coefficients]));
                    }};else{{ // 不添加截距项
                            result_Ridge=ridgeCV(reg_df,`R,factor_list,alphas={self.Ridge_lamdas},intercept=false);
                            result_Ridge=dict(result_Ridge[`xColNames],result_Ridge[`coefficients]);
                    }};
                    beta_df=table(result_Ridge.keys() as indicator,result_Ridge.values() as value);
                    beta_df=select "R_Ridge" as class,* from beta_df;
                    summary_result.append!(beta_df);
                    }};
                    
                    // 统计结果(Multisummary_reuslt,ElasticNet)
                    if (ElasticNet_estimation==1){{
                        if (add_Intercept==1){{  // 添加截距项
                            result_ElasticNet=elasticNetCV(reg_df,`R,factor_list,alphas={self.ElasticNet_lamdas},intercept=true);
                            result_ElasticNet=dict([`intercept].append!(result_ElasticNet[`xColNames]),[result_ElasticNet[`intercept]].append!(result_ElasticNet[`coefficients]));
                    }};else{{ // 不添加截距项
                            result_ElasticNet=elasticNetCV(reg_df,`R,factor_list,alphas={self.ElasticNet_lamdas},intercept=false);
                            result_ElasticNet=dict(result_ElasticNet[`xColNames],result_ElasticNet[`coefficients]);
                    }};
                    beta_df=table(result_ElasticNet.keys() as indicator,result_ElasticNet.values() as value);
                    beta_df=select "R_ElasticNet" as class,* from beta_df;
                    summary_result.append!(beta_df);
                    }};
                    
                    summary_result.append!(IC_df);
                    summary_result.append!(RankIC_df);
                    summary_result=select p as period,* from summary_result;  // 最后添加日期
                    if (counter==0){{
                        final_pos_result=summary_result.copy();}};
                    else{{
                        final_pos_result.append!(summary_result);}};
                    counter=counter+1;
                }} // reg_df元素>0 END
                else{{}}
            }};   
            undef(`summary_result); // 释放内存
                
            // 格式调整!!
            final_pos_result=select benchmark_str as Benchmark,period,class,indicator,value from final_pos_result;
            loadTable('{self.result_database}','{self.Multisummary_table}').append!(final_pos_result);
            undef(`final_pos_result); // 释放内存
        }};
        undef(`pt`reg_df); //释放内存
        """

    def individual_command(self):
        """合并资产区间real_return+区间return_pred"""
        return rf"""
        benchmark_str="{self.benchmark}";
        factor_list={self.factor_list};
        Ridge_estimation=int({int(self.Ridge_estimation)});  // 是否估计Ridge因子收益率
        Lasso_estimation=int({int(self.Lasso_estimation)});  // 是否估计Lasso因子收益率
        ElasticNet_estimation=int({int(self.ElasticNet_estimation)});   // 是否估计ElasticNet因子收益率
        
        // 个股因子值+区间收益率数据
        basic_select=["Benchmark","period","symbol","real_return"];
        current_select=factor_list.copy();
        slice_pt=sql(select=sqlCol(basic_select.copy().append!(current_select)),from=loadTable("{self.result_database}","{self.individualF_table}"),where=[<Benchmark=benchmark_str>]).eval();
        undef(`individual_pt); // 释放内存
        for (col in factor_list){{
            rename!(slice_pt,col,string(col)+"_factor_value");
        }};
    
        // 因子收益率数据
        factor_pt=select * from loadTable("{self.result_database}","{self.factorR_table}") where Benchmark=benchmark_str and class in ["R_OLS","R_Lasso","R_Ridge","R_ElasticNet"];
        rename!(factor_pt,`value,`factor_return);
        rename!(factor_pt,`value_pred,`factor_return_pred);
        pred_factor_OLS=select factor_return_pred from factor_pt where class="R_OLS" pivot by period,indicator; // 预测因子收益率(OLS)
        
        // OLS预期收益率计算
        OLS_pt=select * from slice_pt left join pred_factor_OLS on pred_factor_OLS.period=slice_pt.period;
        OLS_pt[`method]="OLS"; // 表示估计方法为OLS的因子收益率得到的个股收益率预测值
        undef(`pred_factor_OLS); // 内存释放
        for (col in factor_list){{
            OLS_pt[string(col)+"_return_pred"]=OLS_pt[col+"_alpha"]+OLS_pt[col+"_factor_value"]*OLS_pt[col]; // α+因子值*β
            dropColumns!(OLS_pt,string(col)+"_factor_value"); // 删了因子值
            dropColumns!(OLS_pt,string(col)+"_alpha");    // 删了alpha
            dropColumns!(OLS_pt,string(col)); // 删了β
        }};
    
        // Lasso预期收益率计算
        if (Lasso_estimation==1){{
            pred_factor_Lasso=select factor_return_pred from factor_pt where class="R_Lasso" pivot by period,indicator; // 预测因子收益率(Lasso)
            Lasso_pt=select * from slice_pt left join pred_factor_Lasso on pred_factor_Lasso.period=slice_pt.period;
            Lasso_pt[`method]="Lasso"; // 表示估计方法为OLS的因子收益率得到的个股收益率预测值
            undef(`pred_factor_Lasso); // 内存释放
            for (col in factor_list){{
                Lasso_pt[string(col)+"_return_pred"]=Lasso_pt[col+"_alpha"]+Lasso_pt[col+"_factor_value"]*Lasso_pt[col]; // α+因子值*β
                dropColumns!(Lasso_pt,string(col)+"_factor_value"); // 删了因子值
                dropColumns!(Lasso_pt,string(col)+"_alpha");    // 删了alpha
                dropColumns!(Lasso_pt,string(col)); // 删了β
            }};
            OLS_pt.append!(Lasso_pt);  // 合并数据
            undef(`Lasso_pt); // 内存释放
        }};
        
        // Ridge预期收益率计算
        if (Ridge_estimation==1){{
            pred_factor_Ridge=select factor_return_pred from factor_pt where class="R_Ridge" pivot by period,indicator; // 预测因子收益率(Ridge)
            Ridge_pt=select * from slice_pt left join pred_factor_Ridge on pred_factor_Ridge.period=slice_pt.period;
            Ridge_pt[`method]="Ridge"; // 表示估计方法为OLS的因子收益率得到的个股收益率预测值
            undef(`pred_factor_Ridge); // 内存释放
            for (col in factor_list){{
                Ridge_pt[string(col)+"_return_pred"]=Ridge_pt[col+"_alpha"]+Ridge_pt[col+"_factor_value"]*Ridge_pt[col]; // α+因子值*β
                dropColumns!(Ridge_pt,string(col)+"_factor_value"); // 删了因子值
                dropColumns!(Ridge_pt,string(col)+"_alpha");    // 删了alpha
                dropColumns!(Ridge_pt,string(col)); // 删了β
            }};
            OLS_pt.append!(Ridge_pt);  // 合并数据
            undef(`Ridge_pt); // 内存释放
        }};
        
        // ElasticNet预期收益率计算
        if (ElasticNet_estimation==1){{
            pred_factor_ElasticNet=select factor_return_pred from factor_pt where class="R_ElasticNet" pivot by period,indicator; // 预测因子收益率(ElasticNet)
            ElasticNet_pt=select * from slice_pt left join pred_factor_ElasticNet on pred_factor_ElasticNet.period=slice_pt.period;
            ElasticNet_pt[`method]="ElasticNet"; // 表示估计方法为OLS的因子收益率得到的个股收益率预测值
            undef(`pred_factor_ElasticNet); // 内存释放
            for (col in factor_list){{
                ElasticNet_pt[string(col)+"_return_pred"]=ElasticNet_pt[col+"_alpha"]+ElasticNet_pt[col+"_factor_value"]*ElasticNet_pt[col]; // α+因子值*β
                dropColumns!(ElasticNet_pt,string(col)+"_factor_value"); // 删了因子值
                dropColumns!(ElasticNet_pt,string(col)+"_alpha");    // 删了alpha
                dropColumns!(ElasticNet_pt,string(col)); // 删了β
            }};
            OLS_pt.append!(ElasticNet_pt);  // 合并数据
            undef(`ElasticNet_pt); // 内存释放
        }};
        
        // 添加至数据库
        loadTable("{self.result_database}","{self.individualR_table}").append!(OLS_pt);
        undef(`OLS_pt`slice_pt`factor_pt); // 内存释放
        """

    def MultiIndividual_command(self):
        """[多因子]根据FactorR计算资产的预测收益率"""
        L=[]
        for i,j in zip([self.Lasso_estimation,self.Ridge_estimation,self.ElasticNet_estimation],["Lasso","Ridge","ElasticNet"]):
            if i:
                L.append(j)
        if len(L)==0:   # 说明只估计OLS,需要把最后的,删掉
            MultiString="""
            final_result=select firstNot(Benchmark) as Benchmark,firstNot(real_return) as real_return,factor_value**factor_return_pred_OLS as return_pred_OLS from individual_pt group by period,symbol;
            undef(`individual_pt);
            final_result=select Benchmark,period,symbol,real_return,return_pred_OLS from final_result;
            """
        else:
            MultiString=f"""
            final_result=select firstNot(Benchmark) as Benchmark,firstNot(real_return) as real_return,factor_value**factor_return_pred_OLS as return_pred_OLS,{','.join([f"factor_value**factor_return_pred_{k} as return_pred_{k}" for k in L])} from individual_pt group by period,symbol;
            undef(`individual_pt);
            final_result=select Benchmark,period,symbol,real_return,return_pred_OLS,{','.join([f"return_pred_{k}" for k in L])} from final_result;
            """
        return rf"""
        benchmark_str="{self.benchmark}"
        add_Intercept=int({int(self.Multi_Intercept)}); // 多因子模型是否添加截距项
        Intercept=`intercept;   // DolphinDB SQL ols命令默认的截距项factor name
        factor_list={self.factor_list};
        total_factor_list=factor_list.copy();
        total_factor_list.append!(Intercept); // 添加了截距项的factor_list
        Ridge_estimation=int({int(self.Ridge_estimation)});  // 是否估计Ridge因子收益率
        Lasso_estimation=int({int(self.Lasso_estimation)});  // 是否估计Lasso因子收益率
        ElasticNet_estimation=int({int(self.ElasticNet_estimation)});   // 是否估计ElasticNet因子收益率
        
        // 个股因子值+区间收益率数据
        individual_pt=select * from loadTable("{self.result_database}","{self.individualF_table}") where Benchmark=benchmark_str;
        
        // 添加截距项(DOUBLE format)
        if (add_Intercept==1){{
            individual_pt[Intercept]=1.0;
            individual_pt=unpivot(individual_pt,keyColNames=`Benchmark`period`symbol`real_return,valueColNames=total_factor_list);
        }}else{{
            individual_pt=unpivot(individual_pt,keyColNames=`Benchmark`period`symbol`real_return,valueColNames=factor_list);
        }};
        rename!(individual_pt,`valueType,`indicator);
        rename!(individual_pt,`value,`factor_value);
        
        // 因子收益率数据
        factor_pt=select * from loadTable("{self.result_database}","{self.MultifactorR_table}") where Benchmark=benchmark_str and class in ["R_OLS","R_Lasso","R_Ridge","R_ElasticNet"];
        rename!(factor_pt,`value,`factor_return);
        rename!(factor_pt,`value_pred,`factor_return_pred);
        // dropColumns!(factor_pt,`start_date`end_date);
        
        OLS_pt=select Benchmark,period,indicator,factor_return_pred from factor_pt where class="R_OLS";
        rename!(OLS_pt,`factor_return_pred,`factor_return_pred_OLS); // 预测因子收益率(OLS)
        
        if (Lasso_estimation==1){{
            Lasso_pt=select Benchmark,period,indicator,factor_return_pred from factor_pt where class="R_Lasso";
            rename!(Lasso_pt,`factor_return_pred,`factor_return_pred_Lasso); // 预测因子收益率(Lasso)
        }};
        
        if (Ridge_estimation==1){{
            Ridge_pt=select Benchmark,period,indicator,factor_return_pred from factor_pt where class="R_Ridge";
            rename!(Ridge_pt,`factor_return_pred,`factor_return_pred_Ridge); // 预测因子收益率(Ridge)
        }};
        
        if (ElasticNet_estimation==1){{
            ElasticNet_pt=select Benchmark,period,indicator,factor_return_pred from factor_pt where class="R_ElasticNet";
            rename!(ElasticNet_pt,`factor_return_pred,`factor_return_pred_ElasticNet); // 预测因子收益率(ElasticNet)
        }};
        undef(`factor_pt); // 释放内存
        
        // Combine因子收益率(OLS/Lasso/Ridge/ElasticNet)
        individual_pt=select * from individual_pt left join OLS_pt on individual_pt.Benchmark=OLS_pt.Benchmark and individual_pt.period=OLS_pt.period and OLS_pt.indicator=individual_pt.indicator;
        undef(`OLS_pt);
        
        if (Lasso_estimation==1){{
            individual_pt=select * from individual_pt left join Lasso_pt on individual_pt.Benchmark=Lasso_pt.Benchmark and individual_pt.period=Lasso_pt.period and Lasso_pt.indicator=individual_pt.indicator;
            undef(`Lasso_pt)
        }};
        
        if (Ridge_estimation==1){{
            individual_pt=select * from individual_pt left join Ridge_pt on individual_pt.Benchmark=Ridge_pt.Benchmark and individual_pt.period=Ridge_pt.period and Ridge_pt.indicator=individual_pt.indicator;
            undef(`Ridge_pt);
        }};
        
        if (ElasticNet_estimation==1){{
            individual_pt=select * from individual_pt left join ElasticNet_pt on individual_pt.Benchmark=ElasticNet_pt.Benchmark and individual_pt.period=ElasticNet_pt.period and ElasticNet_pt.indicator=individual_pt.indicator;
            undef(`ElasticNet_pt); 
        }};
        
        // 计算预测收益率(OLS)
        {MultiString}
        loadTable("{self.result_database}","{self.MultiIndividualR_table}").append!(final_result);
        undef(`final_result);
        """

    def OptimizeData_command(self):
        """投资组合优化框架
        [新增]: 由于添加了SingleFactor_estimation &MultiFactor_estimation两个参数,导致只能合并一个表
        当前是SingleFactor Estimation & MultiFactor Estimation两者必选其一的 Model Estimation 可选
        """
        if self.ModelIndividualR_predictFunc:
            Model_state=1
        else:
            Model_state=0
        L=[]
        for i,j in zip([self.Lasso_estimation,self.Ridge_estimation,self.ElasticNet_estimation],["Lasso","Ridge","ElasticNet"]):
            if i:
                L.append(j)
        if len(L)>0:
            OptimizeData_string=f"""
            Multi_pt=select period,symbol,return_pred_OLS as OLS,{",".join([f"return_pred_{i} as {i}" for i in L])} from loadTable("{self.result_database}","{self.MultiIndividualR_table}") where Benchmark=benchmark_str;
            Multi_pt=unpivot(Multi_pt,keyColNames=["period","symbol"],valueColNames=["OLS"].append!({L}));
            rename!(Multi_pt,`period`symbol`method`Multi_return_pred);
            """
            OptimizeData_string2=f"""
            pt=select Benchmark,period,symbol,real_return,return_pred_OLS as OLS,{",".join([f"return_pred_{i} as {i}" for i in L])} from loadTable("{self.result_database}","{self.MultiIndividualR_table}") where Benchmark=benchmark_str;
            pt=unpivot(pt,keyColNames=["Benchmark","period","symbol","real_return"],valueColNames=["OLS"].append!({L}));
            rename!(pt,`Benchmark`period`symbol`real_return`method`Multi_return_pred);
            """
        else:
            OptimizeData_string=f"""
            Multi_pt=select period,symbol,return_pred_OLS as OLS from loadTable("{self.result_database}","{self.MultiIndividualR_table}") where Benchmark=benchmark_str;
            Multi_pt=unpivot(Multi_pt,keyColNames=["period","symbol"],valueColNames=["OLS"]);
            rename!(Multi_pt,`period`symbol`method`Multi_return_pred);
            """
            OptimizeData_string2=f"""
            pt=select Benchmark,period,symbol,real_return,return_pred_OLS as OLS from loadTable("{self.result_database}","{self.MultiIndividualR_table}") where Benchmark=benchmark_str;
            pt=unpivot(pt,keyColNames=["Benchmark","period","symbol","real_return"],valueColNames=["OLS"]);
            rename!(pt,`Benchmark`period`symbol`real_return`method`Multi_return_pred);
            """

        return rf""" 
        benchmark_str="{self.benchmark}";
        Ridge_estimation=int({int(self.Ridge_estimation)});  // 是否估计Ridge因子收益率
        Lasso_estimation=int({int(self.Lasso_estimation)});  // 是否估计Lasso因子收益率
        ElasticNet_estimation=int({int(self.ElasticNet_estimation)});   // 是否估计ElasticNet因子收益率
        
        // Template
        template_pt=select symbol,start_date as date,period from loadTable("{self.combine_database}","{self.template_individual_table}");
        
        // Return Combination
        if (int({int(self.SingleFactor_estimation)})==1){{  // 说明是估计单因子收益率的
            pt=select * from loadTable("{self.result_database}","{self.individualR_table}") where Benchmark=benchmark_str;
            if (int({int(self.MultiFactor_estimation)})==1){{ // 说明是同时估计多因子收益率的
                {OptimizeData_string}
                // 合并单因子和多因子return结果
                pt=select * from pt left join Multi_pt on pt.period=Multi_pt.period and Multi_pt.symbol=pt.symbol and pt.method=Multi_pt.method;
                undef(`Multi_pt);
            }};
            else{{}};  // 说明只估计单因子收益率
        }};
        else{{ // 说明只估计多因子收益率
            {OptimizeData_string2}
        }};
        
        // 合并模型return结果
        Model_state=int({Model_state});
        if (Model_state==1){{  // 说明需要合并模型的结果
            Model_pt=select * from loadTable("{self.result_database}","{self.ModelIndividualR_table}") where Benchmark=benchmark_str;
            dropColumns!(Model_pt,`Benchmark`real_return);
            pt=select * from pt left join Model_pt on pt.period=Model_pt.period and pt.symbol=Model_pt.symbol;
            undef(`Model_pt);
        }};else{{}};
        
        // 个股info symbol date marketvalue industry
        info_pt=select firstNot(marketvalue) as marketvalue,first(industry) as industry from loadTable("{self.combine_database}","{self.combine_table}") group by symbol,date;
        pt=select * from pt left join template_pt on template_pt.symbol=pt.symbol and template_pt.period=pt.period; // 从Template中添加date列
        pt=select * from pt left join info_pt on info_pt.symbol=pt.symbol and info_pt.date=pt.date;
        dropColumns!(pt,`date);
        undef(`info_pt); // 释放内存
        
        update pt set industry=nullFill(industry.ffill!(),"NA") context by symbol;  // 填充空缺值行业
        asset_pt=select period,symbol,{','.join(self.Group_list)} from loadTable("{self.result_database}","{self.asset_slice_table}") where Benchmark=benchmark_str;

        // 合并信号值
        pt=select * from pt left join asset_pt on asset_pt.symbol=pt.symbol and asset_pt.period=pt.period;
        undef(`asset_pt); //释放内存

        // 上传至数据库
        loadTable("{self.optimize_database}","{self.optimize_data_table}").append!(pt);
        undef(`pt`info_pt);  // 释放内存 
        """

    def BackTest(self):
        # 因子模型部分
        self.init_ResultDataBase(dropDatabase=True)
        for benchmark in tqdm.tqdm(self.benchmark_list,desc=f"Calculating SingleFactor result"):
            self.benchmark=benchmark
            self.session.run(self.summary_command()) # Step1.向量化生成单因子回测的IC/RankIC/等可向量化运算指标(即使采用WalkForward框架估计也能得到相同的内容的部分)
            # summary_command计算数据的功能,因而必须执行,单因子测试的部分在执行命令时判断,MultiSummary_command则没有这个功能
            if self.SingleFactor_estimation:
                 self.pred_FactorR()                      # Step1.5 [自定义] 利用t-1期WalkForward分别对t期的因子收益率+alpha进行预测，结合t期已知的因子值，给出个股预测的收益率
                 self.session.run(self.individual_command()) # Step2. 根据FactorR(单因子)代入模型得到个股预期收益率expect_return,返回symbol date period start_ts end_ts expect_return real_return return_pred
        if self.MultiFactor_estimation:     # 说明需要估计多因子收益率
            for benchmark in tqdm.tqdm(self.benchmark_list,desc=f"Calculating MultiFactor result"):
                self.benchmark=benchmark
                self.session.run(self.Multisummary_command()) # Step1.向量化生成单因子回测的IC/RankIC/等可向量化运算指标(即使采用WalkForward框架估计也能得到相同的内容的部分)
                self.pred_MultiFactorR()                      # Step1.5 [自定义] 利用t-1期WalkForward分别对t期的因子收益率+alpha进行预测，结合t期已知的因子值，给出个股预测的收益率
                self.session.run(self.MultiIndividual_command()) # Step2.根据MultiFactorR代入模型得到个股预期收益率expect_return

    def ModelTest(self):
        # ML/DL模型部分
        if self.ModelIndividualR_predictFunc:   # 说明传入了自定义函数进行ModelPredict
            # Init ModelTable()
            self.init_ModelDatabase()
            for benchmark in tqdm.tqdm(self.benchmark_list,desc=f"Calculating ModelReturn result"):
                self.benchmark=benchmark
                self.pred_ModelIndividualR()    # Step2. [自定义] 应用其他ML/DL模型预测资产预期收益率

    def Slice(self):
        # 因子&资产选择部分
        self.init_SliceDatabase()
        for benchmark in tqdm.tqdm(self.benchmark_list,desc=f"Slicing Factor&Asset"):
            self.benchmark=benchmark
            if self.Factor_sliceFunc:
                self.Factor_sliceFunc(self)
            if self.Asset_sliceFunc:
                self.Asset_sliceFunc(self) # Step3. [自定义]根据因子IC/RankIC/其他指标选择因子,再根据所选因子对个股构建等权组合

    def Optimize(self):
        # 组合优化部分
        self.init_OptimizeDatabase(dropDatabase=True)    # init_optimize_database
        for benchmark in tqdm.tqdm(self.benchmark_list,desc=f"Optimizing result"):
            self.benchmark=benchmark
            self.session.run(self.OptimizeData_command()) # 准备投资组合优化的Data
            self.OptimizeFunc(self)         # Step4. [自定义]根据个股预期收益率+其他约束条件+(可选:风险模型)构建优化器优化该组合


if __name__=="__main__":
    session=ddb.session()
    session.connect("localhost",8848,"admin","123456")
    pool=ddb.DBConnectionPool("localhost",8848,10,"admin","123456")
    from factor_func.Data_func import ReturnModel_Data as R
    from factor_func.ReturnModel_func import *
    from factor_func.Optimize_func_riskfolio import *
    from factor_func.Model import *
    F=ReturnModel_Backtest(
        # 基准信息
        session=session,pool=pool,start_date="2020.01.04",end_date="2025.01.27",posPeriod=5,
        # 数据准备（行情数据(函数)+基准指数(函数)+因子数据(函数)→合并数据(函数)）
        symbol_database="dfs://stock_cn/ReturnModel",symbol_table="symbol",Symbol_prepareFunc=R.prepare_symbol_data,
        benchmark_database="dfs://stock_cn/ReturnModel",benchmark_table="benchmark",benchmark_list=["b000985"],Benchmark_prepareFunc=R.prepare_benchmark_data,
        factor_database="dfs://stock_cn/factor",factor_table="factor",factor_list=["Quantum10","Quantum5","Volume10","Volume5"],Factor_prepareFunc=R.prepare_factor_data,
        combine_database="dfs://stock_cn/ReturnModel",combine_table="combination",Combine_prepareFunc=R.prepare_combine_data,
        result_database="dfs://stock_cn/ReturnModel_result",period_return_Algo="""
        nullFill(
            (lastNot(close)-firstNot(open))/firstNot(open)-(lastNot(benchmark_close)-firstNot(benchmark_open))/firstNot(benchmark_open),
        0) context by period_symbol
        """,# period return(一个period内所有时刻共享)
        daily_return_Algo="""
        nullFill(
            (next_open-open)/open-(next_benchmark_open-benchmark_open)/benchmark_open,
        0)
        """,# period→next period(一个period的时刻顺序对应下一个period的时刻)
        SingleFactor_estimation=True,MultiFactor_estimation=True,Multi_Intercept=True,
        Lasso_estimation=False,Ridge_estimation=True,ElatsicNet_estimation=False,
        Ridge_lamdas=[0.001,0.005,0.01,0.05,0.1,0.5,1,5,10,50,100,500,1000],
        Lasso_lamdas=[0.001,0.005,0.01,0.05,0.1,0.5,1,5,10,50,100,500,1000],
        ElasticNet_lamdas=[0.001,0.005,0.01,0.05,0.1,0.5,1,5,10,50,100,500,1000],
        # 因子收益率预测方法(自定义,必须)
        FactorR_predictFunc=FactorR_pred,MultiFactorR_predictFunc=MultiFactorR_pred,
        # 模型收益率预测方法(自定义,可选)
        # ModelR_predictFunc=ModelBackTest,Model_list=["RandomForest","AdaBoost"],
        # 因子选择&资产选择方法(自定义,可选)   # Composition(自定义):按IC加权合成因子值分十组
        Factor_sliceFunc=Factor_slice,Asset_sliceFunc=Stock_slice,Group_list=["Composition","MV","IND"],
        # 投资组合优化结果保存(自定义,可选)
        optimize_database="dfs://stock_cn/ReturnModel_optimize",Optimize_func=execute_optimize,
        optstrategy_list=["TOP30","TOP30HRP","TOP30_industry","TOP30HRP_sharpe",
                         "TOP50","TOP50HRP","TOP50_industry","TOP50HRP_sharpe"]
    )
    # F.init_SymbolDatabase(dropDatabase=True)
    # F.init_BenchmarkDatabase()
    # F.init_FactorDatabase(dropDatabase=True)
    # F.add_SymbolData()
    # F.add_BenchmarkData()
    # F.add_FactorData()

    # 如果原始数据没有变化，那么不用运行init_CombineDatabase()与add_CombineData()
    # F.init_CombineDataBase()
    # F.add_CombineData()

    # F.BackTest()
    # F.ModelTest()
    # F.Slice()
    F.Optimize()