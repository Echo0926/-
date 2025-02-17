import os,sys
import time
import pandas as pd
import numpy as np
import dolphindb as ddb
import tqdm
import matplotlib.pyplot as plt
from basic import *
import warnings
warnings.filterwarnings("ignore")
pd.set_option("display.max_columns",None)
plt.rcParams['font.sans-serif']=['KaiTi'] # 显示中文
plt.rcParams['axes.unicode_minus']=False # 显示负号

class Risk_Backtest:
    """
    多因子模型风险模型
    回归模型：r(t-t+1)=α+β(r_benchmark)
    """
    def __init__(self,session,pool,start_date,end_date,
                 symbol_database,symbol_table,Symbol_prepareFunc,    # 标的数据库(后续市值数据也可以加进去)
                 factor_database,factor_table,factor_list,Factor_prepareFunc,          # 因子数据库
                 benchmark_database,benchmark_table,benchmark_list,Benchmark_prepareFunc, # 基准收益数据库
                 combine_database,combine_table,Combine_prepareFunc,    # 合并(标的+因子+基准收益)数据库
                 Factorresult_database,Assetresult_pathdir=None, # 资产收益率协方差矩阵保存方式(【新增:取消线上保存方式】强制保存在本地)
                 industry_list=None,Industry_prepareFunc=None,  # 行业数据
                 posPeriod=[5]):
        """
        symbol_database: symbol date price industry state
        factor_database: symbol date factor_name factor_value
        benchmark_database: symbol date benchmark_price
        """
        # 基本信息
        self.strategy_name="CoVar_Risk"
        self.session=session
        self.pool=pool

        # 库表类
        self.symbol_database=symbol_database
        self.symbol_table=symbol_table
        self.factor_database=factor_database
        self.factor_table=factor_table
        self.benchmark_database=benchmark_database
        self.benchmark_table=benchmark_table
        self.benchmark_list=benchmark_list
        self.combine_database=combine_database  # symbol date
        self.combine_table=combine_table
        self.result_database=Factorresult_database  # (简写为result_database,因为本意为因子层面的协方差收益率矩阵计算)

        # 变量类
        if isinstance(posPeriod,int) or isinstance(posPeriod,float):
            self.posPeriod=[posPeriod]    # 调仓周期
        self.t=self.posPeriod[0]    # 持仓时间
        self.benchmark=None         # 基准标的
        self.start_date=start_date
        self.end_date=end_date
        self.start_dot_date=pd.Timestamp(self.start_date).strftime('%Y.%m.%d')
        self.end_dot_date=pd.Timestamp(self.end_date).strftime('%Y.%m.%d')
        self.factor_list=factor_list
        if not industry_list:
            self.industry_list=[]
        else:
            self.industry_list=industry_list
        self.total_period_list=None
        self.current_date=None
        self.current_period=None

        # 函数类
        self.Symbol_prepareFunc=Symbol_prepareFunc
        self.Benchmark_prepareFunc=Benchmark_prepareFunc
        self.Factor_prepareFunc=Factor_prepareFunc
        self.Industry_prepareFunc=Industry_prepareFunc
        self.Combine_prepareFunc=Combine_prepareFunc

        # 中间计算+最终结果类
        self.template_table="template"                  # 包含 start_date end_date period,为回测结果的基石
        self.template_daily_table="template_daily"      # 包含 symbol date 两列
        self.template_individual_table="template_individual"    # 包含 symbol start_date end_date period,为回测的所有结果

        self.individualF_table="individual" # 每个区间的因子值
        self.individualF_daily_table="individual_daily"   # 每天的因子值不需要存储

        self.summary_table="summary"
        self.summary_daily_table="summary_daily"
        self.individualR_table="individual_return"
        self.individual_daily_table="individual_result_daily"
        self.factorCov_table="factor_Covar"
        self.uniqueCov_table="unique_Covar"

        # 其他
        self.Assetresult_pathdir=Assetresult_pathdir    #

    def init_SymbolDatabase(self):
        """[Optional]第一次运行,初始化行情数据库"""
        if not self.session.existsTable(dbUrl=self.symbol_database,tableName=self.symbol_table):
            self.session.run(f"""
            db=database("{self.symbol_database}",RANGE,2000.01M+(0..30)*12,engine="TSDB")
            schemaTb=table(1:0,`symbol`date`price`marketvalue`state,[SYMBOL,DATE,DOUBLE,DOUBLE,DOUBLE]);
            t=db.createPartitionedTable(table=schemaTb,tableName="{self.symbol_table}",partitionColumns="date",sortColumns=["symbol","date"],keepDuplicates=LAST)
            """)
        else:
            pass

    def init_BenchmarkDatabase(self):
        """[Optional]第一次运行,初始化基准收益数据库"""
        if not self.session.existsTable(dbUrl=self.benchmark_database,tableName=self.benchmark_table):
            self.session.run(f"""
            db=database("{self.benchmark_database}",RANGE,2000.01M+(0..30)*12,engine="TSDB")
            schemaTb=table(1:0,`symbol`date`price,[SYMBOL,DATE,DOUBLE]);
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
        """[Necessary]初始化合并数据库"""
        # 默认每次回测前删除上次的因子库（因为因子个数和名称可能不一样）
        if self.session.existsTable(dbUrl=self.combine_database,tableName=self.combine_table):
            self.session.dropTable(dbPath=self.combine_database,tableName=self.combine_table)
        columns_name=["symbol","date","price","marketvalue","state"]+["period"]+self.benchmark_list+self.factor_list+self.industry_list
        columns_type=["SYMBOL","DATE","DOUBLE","DOUBLE","DOUBLE"]+["DOUBLE"]+["DOUBLE"]*len(self.benchmark_list)+["DOUBLE"]*len(self.factor_list)+["DOUBLE"]*len(self.industry_list)
        self.session.run(f"""
        db=database("{self.combine_database}",RANGE,2000.01M+(0..30)*12,engine="TSDB")
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
        """[Necessary]
        Barra风险模型结果输出表(因子层面)
        """
        # individual_result (period_Xvalue(区间因子值),period_return(当前区间收益率),next_period_return(下一个区间收益率))
        if dropDatabase:
            if session.existsDatabase(dbUrl=self.result_database):  # 删除数据库
                self.session.dropDatabase(dbPath=self.result_database)
            else:
                pass
        if self.session.existsTable(dbUrl=self.result_database,tableName=self.individualF_table):
            self.session.dropTable(dbPath=self.result_database,tableName=self.individualF_table)
        columns_name=["Benchmark","period","symbol"]+self.factor_list+["real_return"]
        columns_type=["SYMBOL","DOUBLE","SYMBOL"]+["DOUBLE"]*len(self.factor_list)+["DOUBLE"]
        self.session.run(f"""
        // db1=database("dfs://db1",RANGE,2000.01M+(0..30)*12,engine="TSDB"); // 按照时间分区
        // db2=database("dfs://db2",LIST,{self.benchmark_list},engine="TSDB"); // 按照benchmark_list分区
        // db=database("{self.result_database}",COMPO,[db1,db2],engine="TSDB");
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

        # IndividualR_result(相比IndividualR_result少了一列indicator,因为是所有因子合在一起预测的结果)
        if self.session.existsTable(dbUrl=self.result_database,tableName=self.individualR_table):
            self.session.dropTable(dbPath=self.result_database,tableName=self.individualR_table)
        columns_name=["Benchmark","period","symbol","real_return"]+["expect_return_OLS","expect_return_Lasso","expect_return_Ridge","expect_return_ElasticNet"]
        columns_type=["SYMBOL","DOUBLE","SYMBOL","DOUBLE"]+["DOUBLE"]*4
        self.session.run(f"""
        db=database("{self.result_database}");
        schemaTb=table(1:0,{columns_name},{columns_type});
        t=db.createPartitionedTable(table=schemaTb,tableName="{self.individualR_table}",partitionColumns=["Benchmark"])
        """)

        # factor_cov(因子收益率协方差矩阵,调仓日拼接)
        if self.session.existsTable(dbUrl=self.result_database,tableName=self.factorCov_table):
            self.session.dropTable(dbPath=self.result_database,tableName=self.factorCov_table)
        columns_name=["Benchmark","date","F1","F2","FactorR_Cov"]
        columns_type=["SYMBOL","DATE","SYMBOL","SYMBOL","DOUBLE"]
        self.session.run(f"""
        db=database("{self.result_database}");
        schemaTb=table(1:0,{columns_name},{columns_type});
        t=db.createPartitionedTable(table=schemaTb,tableName="{self.factorCov_table}",partitionColumns=["Benchmark"])
        """)

    def add_SymbolData(self):
        self.Symbol_prepareFunc(self)

    def add_BenchmarkData(self):
        self.Benchmark_prepareFunc(self)

    def add_FactorData(self):
        self.Factor_prepareFunc(self)

    def add_IndustryData(self):
        self.Industry_prepareFunc(self)

    def add_CombineData(self):
        self.Combine_prepareFunc(self)

    def summary_command(self):
        """individual_return(period_return)&summary_result&summary_daily_result
        【新增】:取消了IC/RankIC的统计,仅保留了OLS估计部分,从而加快了运算速度
        """
        return rf"""
        // 多因子回测框架
        pt=select * from loadTable("{self.combine_database}","{self.combine_table}");
        factor_list={self.factor_list}; // 因子列表
        
        // 在不同的benchmark+period下计算预期收益率
        for (benchmark_str in ["{self.benchmark}"]){{    
            //【0.Preparation】
            pt[`benchmark]=pt[benchmark_str];
    
            //【1.PosOLS】 【挣扎的点:时间点的选择→firstNot or lastNot,其实这个取决于你的因子库是长什么样子的,但还是lastNot好一点其实,因为大部分因子是很难在第一天早上就得出的】
            pt[`辅助列]=string(pt[`period])+pt[`symbol];
            update pt set R=(lastNot(price)-firstNot(price))/firstNot(price)-(lastNot(benchmark)-firstNot(benchmark))/firstNot(benchmark) context by 辅助列;
            pos_pt=select firstNot(symbol) as symbol,firstNot(period) as period, firstNot(R) as R,{','.join(f"first({item}) as {item}" for item in self.factor_list)} from pt group by 辅助列;
            // pos_pt=select * from template_pt left join pos_pt on template_pt.symbol=pos_pt.symbol and template_pt.period=pos_pt.period;
            // pos_pt=select symbol,start_date,end_date,period,R,{','.join(self.factor_list)} from pos_pt;
            sortBy!(pos_pt,`symbol`period,[1,1]); // 按升序排序
                
            // 1.1.Individual结果 
            // [新增]individual_return保存
            individual_return=pos_pt.copy();    // 复制一份,以免和下面summary结果计算部分冲突
            // individual_return[`period_new]="period"+string(individual_return[`period]);
            individual_return=select benchmark_str as Benchmark,period,symbol,{','.join(self.factor_list)},R as period_return from individual_return;
            loadTable('{self.result_database}','{self.individualF_table}').append!(individual_return);
            undef(`individual_return);
                 
            // 1.2.Summary结果
            COUNTER=0;
            distinct_period_list=sort(exec distinct(period) from pos_pt,true);
            for (p in distinct_period_list[:count(distinct_period_list)-1]){{  // 最后一个period肯定没有下一个period的数据
                // Data
                reg_df=select * from pos_pt where period=p and not isNull(R); // 【新增】去除了收益率空缺值的样本进行回归
                // start_date,end_date=mode(reg_df[`start_date]),mode(reg_df[`end_date]);
                    
                // IC&RankIC
                // IC=[];
                // RankIC=[];
                // for (col in factor_list){{
                    // append!(IC,corr(reg_df[col],reg_df[`R]));
                    // append!(RankIC,spearmanr(reg_df[col],reg_df[`R]));
                // }};
                // IC_df=table(factor_list as `indicator,IC as `value);
                // IC_df=select `IC as class, indicator, value from IC_df;
                // RankIC_df=table(factor_list as `indicator,RankIC as `value);
                // RankIC_df=select `RankIC as class, indicator, value from RankIC_df;
                
                //OLS
                result_OLS=ols(reg_df[`R],table(reg_df[factor_list]),intercept=false,mode=2);  // OLS回归结果
                // result_Lasso=lassoBasic(reg_df[`R],table(reg_df[factor_list]),intercept=false,mode=2);  // Lasso回归结果
                result_Lasso=result_OLS.copy(); // 【临时处理】
                // result_Ridge=ridgeBasic(reg_df[`R],table(reg_df[factor_list]),intercept=false,mode=2);  // Ridge回归结果
                result_Ridge=result_OLS.copy(); // 【临时处理】
                //result_ElasticNet=elasticNetBasic(reg_df[`R],table(reg_df[factor_list]),intercept=false,mode=2);  // ElasticNet回归结果
                result_ElasticNet=result_OLS.copy(); // 【临时处理】
                        
                // 统计结果(summary_result,OLS)
                beta_df=select "R_OLS" as class,factor as indicator,beta as value from result_OLS[`Coefficient];
                tstat_df=select "tstat_OLS" as class,factor as indicator,tstat as value from result_OLS[`Coefficient];
                RegDict=dict(result_OLS[`RegressionStat][`item],result_OLS[`RegressionStat][`statistics]);
                R_square=RegDict[`R2];
                Adj_square=RegDict[`AdjustedR2];
                Std_error=RegDict[`StdError];
                Obs=RegDict['Observations'];
                    
                // 添加至summary_table的数据行
                summary_result=table([`R_square_OLS,`Adj_square_OLS,`Std_Error_OLS,`Obs_OLS] as `class, [`R_square,`Adj_square,`Std_Error,`Obs] as `indicator,[R_square,Adj_square,Std_error,Obs] as `value);
                summary_result.append!(beta_df);
                summary_result.append!(tstat_df);
                        
                // 统计结果(summary_result,Lasso)
                beta_df=select "R_Lasso" as class,factor as indicator,beta as value from result_Lasso[`Coefficient];
                tstat_df=select "tstat_Lasso" as class,factor as indicator,tstat as value from result_Lasso[`Coefficient];
                RegDict=dict(result_Lasso[`RegressionStat][`item],result_Lasso[`RegressionStat][`statistics]);
                R_square=RegDict[`R2];
                Adj_square=RegDict[`AdjustedR2];
                Std_error=RegDict[`StdError];
                Obs=RegDict['Observations'];
                summary_result.append!(table([`R_square_Lasso,`Adj_square_Lasso,`Std_Error_Lasso,`Obs_Lasso] as `class, [`R_square,`Adj_square,`Std_Error,`Obs] as `indicator,[R_square,Adj_square,Std_error,Obs] as `value));
                summary_result.append!(beta_df);
                summary_result.append!(tstat_df);
                        
                // 统计结果(summary_result,Ridge)
                beta_df=select "R_Ridge" as class,factor as indicator,beta as value from result_Ridge[`Coefficient];
                tstat_df=select "tstat_Ridge" as class,factor as indicator,tstat as value from result_Ridge[`Coefficient];
                RegDict=dict(result_Ridge[`RegressionStat][`item],result_Ridge[`RegressionStat][`statistics]);
                R_square=RegDict[`R2];
                Adj_square=RegDict[`AdjustedR2];
                Std_error=RegDict[`StdError];
                Obs=RegDict['Observations'];
                summary_result.append!(table([`R_square_Ridge,`Adj_square_Ridge,`Std_Error_Ridge,`Obs_Ridge] as `class, [`R_square,`Adj_square,`Std_Error,`Obs] as `indicator,[R_square,Adj_square,Std_error,Obs] as `value));                        
                summary_result.append!(beta_df);
                summary_result.append!(tstat_df);
                        
                // 统计结果(summary_result,ElasticNet)
                beta_df=select "R_ElasticNet" as class,factor as indicator,beta as value from result_ElasticNet[`Coefficient];
                tstat_df=select "tstat_ElasticNet" as class,factor as indicator,tstat as value from result_ElasticNet[`Coefficient];
                RegDict=dict(result_ElasticNet[`RegressionStat][`item],result_ElasticNet[`RegressionStat][`statistics]);
                R_square=RegDict[`R2];
                Adj_square=RegDict[`AdjustedR2];
                Std_error=RegDict[`StdError];
                Obs=RegDict['Observations'];
                summary_result.append!(table([`R_square_ElasticNet,`Adj_square_ElasticNet,`Std_Error_ElasticNet,`Obs_ElasticNet] as `class, [`R_square,`Adj_square,`Std_Error,`Obs] as `indicator,[R_square,Adj_square,Std_error,Obs] as `value));                        
                summary_result.append!(beta_df);
                summary_result.append!(tstat_df);
                
                // 格式调整
                summary_result=select p as period,* from summary_result;  // 最后添加period
                if (COUNTER==0){{
                    final_pos_result=summary_result.copy();}};
                else{{
                    final_pos_result.append!(summary_result);}};
                COUNTER=COUNTER+1;
            }};
            undef(`pos_pt`reg_df); // 内存释放
                
            // 格式调整!!
            // final_pos_result[`period_new]="period"+string(final_pos_result[`period]);
            final_pos_result=select benchmark_str as Benchmark,period,class,indicator,value from final_pos_result;
            loadTable('{self.result_database}','{self.summary_table}').append!(final_pos_result);
            undef(`final_pos_result); // 内存释放
                    
            // 【2.DailyOLS】
            pt[`辅助列]=string(pt[`symbol])+"period"+string(pt[`period]);
            update pt set dateidx=cumcount(date) context by 辅助列;  // dateidx表示每个资产每个周期内date的标号,根据标号left join price与next_price，从而得到日频的period_return
            next_pt=select period-1 as period,symbol,dateidx,price as next_price,benchmark as next_benchmark from pt;  // 表示下一期period的price(next_price)
            pt=select * from pt left join next_pt on pt.period=next_pt.period and pt.dateidx=next_pt.dateidx and pt.symbol=next_pt.symbol; 
            undef(`next_pt);
            update pt set price=price.ffill() context by 辅助列;
            update pt set next_price=next_price.ffill() context by 辅助列;
            update pt set benchmark=benchmark.ffill() context by 辅助列;   // 是否应该这么写还需要商榷？
            update pt set benchmark=next_benchmark.ffill() context by 辅助列; // 是否应该这么写有待商榷？
            update pt set DailyR=nullFill((next_price-price)/price-(next_benchmark-benchmark)/benchmark,0);
            dropColumns!(pt,`dateidx`price`next_price`benchmark`next_benchmark`辅助列)
            daily_pt=select first(DailyR) as R,{','.join(f"firstNot({item}) as {item}" for item in self.factor_list)} from pt group by symbol,date;
            sortBy!(daily_pt,`symbol`date,[1,1]); // 按升序排序
                
            // 2.1.Individual结果【注：只有需要ML/DL方法的时候才需要保存这个数据库】 
            // [新增]individual_daily_return保存
            // individual_return=select benchmark_str as Benchmark,date,symbol,{','.join(self.factor_list)},R as period_return from daily_pt;
            // loadTable('{self.result_database}','{self.individualF_daily_table}').append!(individual_return);
            // undef(`individual_return);
                
            // 结果列
            COUNTER=0;
            distinct_date_list=sort(exec distinct(date) from daily_pt,true);
            for (t in distinct_date_list[:count(distinct_date_list)-1]){{  // 最后一个date肯定没有下一个date的数据
                // Data
                reg_df=select * from daily_pt where date=t and not isNull(R); // 【新增】去除了收益率为空的样本
                if (count(reg_df)>0){{  // 因为怕有的date没有数据
                    // IC&RankIC
                    // IC=[];
                    // RankIC=[];
                    // for (col in factor_list){{
                        // append!(IC,corr(reg_df[col],reg_df[`R]));
                        // append!(RankIC,spearmanr(reg_df[col],reg_df[`R]));}};
                    // IC_df=table(factor_list as `indicator,IC as `value);
                    // IC_df=select `IC as class, indicator, value from IC_df;
                    // RankIC_df=table(factor_list as `indicator,RankIC as `value);
                    // RankIC_df=select `RankIC as class, indicator, value from RankIC_df;
                
                    //OLS
                    result_df=ols(reg_df[`R],reg_df[factor_list],false,2);

                    // 统计结果(summary_result)
                    beta_df=select "R" as class,factor as indicator,beta as value from result_df[`Coefficient];
                    tstat_df=select "tstat" as class,factor as indicator,tstat as value from result_df[`Coefficient];
                    RegDict=dict(result_df[`RegressionStat][`item],result_df[`RegressionStat][`statistics]);
                    R_square=RegDict[`R2];
                    Adj_square=RegDict[`AdjustedR2];
                    Std_error=RegDict[`StdError];
                    Obs=RegDict['Observations'];
                
                    // 添加至summary_table的数据行
                    summary_result=table([`R_square,`Adj_square,`Std_Error,`Obs] as `class, [`R_square,`Adj_square,`Std_Error,`Obs] as `indicator,[R_square,Adj_square,Std_error,Obs] as `value);
                    summary_result.append!(table([`R_square,`Adj_square,`Std_Error,`Obs] as `class, [`R_square,`Adj_square,`Std_Error,`Obs] as `indicator,[R_square,Adj_square,Std_error,Obs] as `value));
                    summary_result.append!(beta_df);
                    summary_result.append!(tstat_df);
                    
                    // 格式调整
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
        }};
        undef(`pt`template_pt`template_daily_pt); //释放内存
        """

    def MultiIndividual_command(self):
        """[多因子]根据FactorR计算资产的预期收益率"""
        return rf"""
        benchmark_str="{self.benchmark}"
        Intercept=`intercept;   // DolphinDB SQL ols命令默认的截距项factor name
        factor_list={self.factor_list};
        total_factor_list=factor_list.copy();
        total_factor_list.append!(Intercept); // 添加了截距项的factor_list
        
        // 个股因子值+区间收益率数据
        individual_pt=select * from loadTable("{self.result_database}","{self.individualF_table}") where Benchmark=benchmark_str;
        
        // 添加截距项(DOUBLE format)
        individual_pt[Intercept]=1.0;
        individual_pt=unpivot(individual_pt,keyColNames=`Benchmark`period`symbol`real_return,valueColNames=total_factor_list);
        rename!(individual_pt,`valueType,`indicator);
        rename!(individual_pt,`value,`factor_value);
        
        // 因子收益率数据
        factor_pt=select * from loadTable("{self.result_database}","{self.summary_table}") where Benchmark=benchmark_str and class in ["R_OLS","R_Lasso","R_Ridge","R_ElasticNet"];
        rename!(factor_pt,`value,`factor_return);
        // dropColumns!(factor_pt,`start_date`end_date);
        
        OLS_pt=select * from factor_pt where class="R_OLS";
        dropColumns!(OLS_pt,`class);
        rename!(OLS_pt,`factor_return,`factor_return_OLS); // 拟合因子收益率(OLS)
        Lasso_pt=select * from factor_pt where class="R_Lasso";
        dropColumns!(Lasso_pt,`class);
        rename!(Lasso_pt,`factor_return,`factor_return_Lasso); // 拟合因子收益率(Lasso)
        Ridge_pt=select * from factor_pt where class="R_Ridge";
        dropColumns!(Ridge_pt,`class);
        rename!(Ridge_pt,`factor_return,`factor_return_Ridge); // 拟合因子收益率(Ridge)
        ElasticNet_pt=select * from factor_pt where class="R_ElasticNet";
        dropColumns!(ElasticNet_pt,`class);
        rename!(ElasticNet_pt,`factor_return,`factor_return_ElasticNet); // 拟合因子收益率(ElasticNet)
        undef(`factor_pt); // 释放内存
        
        // Combine因子收益率(OLS/Lasso/Ridge/ElasticNet)
        individual_pt=select * from individual_pt left join OLS_pt on individual_pt.Benchmark=OLS_pt.Benchmark and individual_pt.period=OLS_pt.period and OLS_pt.indicator=individual_pt.indicator;
        undef(`OLS_pt);
        individual_pt=select * from individual_pt left join Lasso_pt on individual_pt.Benchmark=Lasso_pt.Benchmark and individual_pt.period=Lasso_pt.period and Lasso_pt.indicator=individual_pt.indicator;
        undef(`Lasso_pt)
        individual_pt=select * from individual_pt left join Ridge_pt on individual_pt.Benchmark=Ridge_pt.Benchmark and individual_pt.period=Ridge_pt.period and Ridge_pt.indicator=individual_pt.indicator;
        undef(`Ridge_pt);
        individual_pt=select * from individual_pt left join ElasticNet_pt on individual_pt.Benchmark=ElasticNet_pt.Benchmark and individual_pt.period=ElasticNet_pt.period and ElasticNet_pt.indicator=individual_pt.indicator;
        undef(`ElasticNet_pt); 
         
        // 计算预测收益率(OLS)
        final_result=select firstNot(Benchmark) as Benchmark,firstNot(real_return) as real_return,factor_value**factor_return_OLS as expect_return_OLS,factor_value**factor_return_Lasso as expect_return_Lasso,factor_value**factor_return_Ridge as expect_return_Ridge,factor_value**factor_return_ElasticNet as expect_return_ElasticNet from individual_pt group by period,symbol;
        undef(`individual_pt);
        final_result=select Benchmark,period,symbol,real_return,return_pred_OLS,return_pred_Lasso,return_pred_Ridge,return_pred_ElasticNet from final_result;
        loadTable("{self.result_database}","{self.individualR_table}").append!(final_result);
        undef(`final_result);
         """

    def factorCov_func(self):
        """
        Step1.指数衰减加权时间序列对于日频协方差序列进行调整
        Step2.Newey-West方法进行调整得到factor_covar_table
        [后续可以继续补充:特征值调整+贝叶斯压缩...]
        Step3.计算得到资产收益率协方差矩阵asset_covar_table
        【注：这里total_period和D均是适用于月度交易日调仓的参数,如果需要更短或更长区间估计FactorReturnCov的话需要修改参数】
        """
        data=self.session.run(rf"""
        // Step1.EWMA估计因子收益率协方差矩阵
        // 假设现在是调仓日
        current_date={self.current_date};
        benchmark_str="{self.benchmark}";
        beta_df=select * from loadTable("{self.result_database}","{self.summary_daily_table}") where class=="R" and date<=current_date;
        indicator_list={self.factor_list}; // 因子list
        total_ts_list=exec distinct(date) as date from beta_df;
        total_ts_list=total_ts_list.sort(false);
        total_period=min(252,count((total_ts_list)));
        current_ts_list=total_ts_list[:total_period];
        D=min(90,total_period);
        beta_matrix=select value from beta_df where indicator in indicator_list and date in current_ts_list pivot by date,indicator; 
        X=beta_matrix.copy();
        mean_df=mean(dropColumns!(X,`date)) // 因子收益率区间均值
        mean_Dict=mean_df[0]; // 因子收益率区间均值Dict
        param_list=[];  // 存储EWMA参数列
        covar_list=[];  // 存储因子收益率CoVar矩阵
        for (i in seq(0,total_period-1)){{
            beta_matrix_slice=select * from beta_matrix where date=current_ts_list[i];
            beta_Dict=beta_matrix_slice[0];
            n=count(indicator_list);
            m=matrix(DOUBLE,n,n);
            for (j in seq(0,n-1)){{
                for (k in seq(0,n-1)){{
                    f1,f2=indicator_list[j],indicator_list[k];
                    m[j,k]=(beta_Dict[f1]-mean_Dict[f1])*(beta_Dict[f2]-mean_Dict[f2]);
                    }};
            }};
            append!(param_list,pow(0.5,double(i)/double(D)));
            append!(covar_list,pow(0.5,double(i)/double(D))*m);
        }};
        EWMA_covar=sum(covar_list)/sum(param_list); //EWMA加权处理
        
        // Step2.Newey-West调整
        // 假设现在是调仓日
        // beta_df=select * from loadTable("{self.result_database}","{self.summary_daily_table}") where class="R" and date<=current_date;  // 上面已经导入了数据,这里直接注释掉
        beta_matrix=select value from beta_df where indicator in indicator_list pivot by date,indicator;
        total_ts_list=exec distinct(date) as date from beta_df;
        total_ts_list=total_ts_list.sort(); // 正序排序[1(上一个调仓周期开始时间),2,...,T-d](T为本次调仓周期开始时间)
        D=min(2,count((total_ts_list)));
        total_period=min(15,count(total_ts_list));  // 这里不一定是15,只是Barra官方给的是15
        bartlett_list=[];  // (外循环)Bartlett权重
        COVAR_list1=[];  // (外循环)Bartlett权重加权
        COVAR_list2=[];
        for (d in seq(1,D)){{
            covar_list1=[]; // (内循环)f*transpose(f)构成的矩阵
            covar_list2=[]; // (内循环)transpose(f)*f构成的矩阵
            param_list=[]; // (内循环)参数权重
            for (t in seq(0,total_period-d-1)){{
            beta_table_t=select * from beta_matrix where date=total_ts_list[t]; // t期的因子收益率矩阵
            dropColumns!(beta_table_t,`date);
            beta_matrix_t=matrix(beta_table_t);
            beta_table_d=select * from beta_matrix where date=total_ts_list[t+d]     // T+d期的因子收益率矩阵
            dropColumns!(beta_table_d,`date);
            beta_matrix_d=matrix(beta_table_d);
            m1=transpose(beta_matrix_d)**beta_matrix_t;
            m2=transpose(m1);
            append!(param_list,pow(0.5,double(total_period-d-t)));
            append!(covar_list1,pow(0.5,double(total_period-d-t))*m1);
            append!(covar_list2,pow(0.5,double(total_period-d-t))*m2);
            }};
            append!(COVAR_list1,(1-double(d)/(double(D)+1))*sum(covar_list1)/sum(param_list));
            append!(COVAR_list2,(1-double(d)/(double(D)+1))*sum(covar_list2)/sum(param_list));
            append!(bartlett_list,(1-double(d)/(double(D)+1)));
        }};

        // 得到最终Newey-west调整后的因子收益率协方差矩阵
        posPeriod=int({self.t});// 当前调仓周期
        final_covar_table=table(posPeriod*(EWMA_covar+sum(COVAR_list1)+sum(COVAR_list2))).rename!(indicator_list);
        // 生成为矩阵格式,后续会用到
        factorR_covar_matrix=matrix(final_covar_table);
        undef(`EWMA_covar); // 释放内存
        undef(`beta_df);
        undef(`beta_matrix);
        
        // 调整格式
        final_covar_table[`factor_name]=indicator_list;
        final_covar_table=unpivot(final_covar_table,keyColNames=`factor_name,valueColNames=columnNames(final_covar_table)[:count(columnNames(final_covar_table))-1]);
        
        // 插入至数据库,此时final_covar_matrix代表了因子收益率的协方差矩阵
        final_covar_table=select benchmark_str as Benchmark,current_date as date,* from final_covar_table;
        loadTable('{self.result_database}','{self.factorCov_table}').append!(final_covar_table);
        undef(`final_covar_table);
        
        // 计算资产协方差风险矩阵
        // 注意:此处indicator_list需要和上面的一致(后续改成sql+indicator_list的格式会好一些)
        X_pt=select symbol,{','.join(self.factor_list)} from loadTable("{self.combine_database}","{self.combine_table}") where date=current_date;
        symbol_list_ori=exec symbol from X_pt; // symbol_list(original)
        symbol_list=["c"]; // 创建一个字符串向量
        for (symbol in symbol_list_ori){{
            symbol_list.append!(string("c")+string(symbol)); // 规范化后的symbol_list(不然不能作列名)
        }};
        symbol_list=symbol_list[1:];
        X_matrix=matrix(X_pt[columnNames(X_pt)[1:]]);
        assetR_covar_table=table(X_matrix**factorR_covar_matrix**transpose(X_matrix)).rename!(symbol_list);
        undef(`factorR_covar_matrix);
        undef(`X_matrix);
        assetR_covar_table=select benchmark_str as Benchmark,current_date as date,symbol_list as symbol,* from assetR_covar_table;
            
        // 保存至本地(取消用DolphinDB保存,现在统一用python保存)
        // saveText(assetR_covar_table,"{self.Assetresult_pathdir}/{self.benchmark}/{str(self.current_date).replace(".","")}.csv",header=true,append=false);
        assetR_covar_table
        """)
        return data

    def uniqueCov_command(self):
        return rf"""
        // Step1.EWMA调整后的特质收益率对角矩阵
        unique_df=select * from loadTable("{self.combine_database}","{self.individual_daily_table}") where date>=date({self.start_uniqueCov_ts}) and date<=date({self.end_uniqueCov_ts});
        unique_matrix=select unique_return from unique_df where class="OLS" pivot by date,symbol;
        code_list=columnNames(unique_matrix)[1:]; // 股票代码(Attention!)
        mean_Dict=mean(unique_matrix)[0];
        total_ts_list=exec distinct(date) as date from unique_matrix;
        total_ts_list=total_ts_list.sort(false); // 时间倒序排序
        total_period=min(252,count((total_ts_list)));
        D=min(90,total_period);
        param_list=[];  // 存储EWMA参数列
        covar_list=[]; // 存储特异收益率协方差矩阵
        for (i in seq(0,total_period-1)){{
            // unique_list=[];  // 存储特异收益率方差序列
            unique_matrix_slice=select * from unique_matrix where date=total_ts_list[i];
            unique_Dict=unique_matrix_slice[0];
            n=count(code_list);
            m=matrix(DOUBLE,n,n);
            for (j in seq(0,n-1)){{
                asset=code_list[j];
                m[j,j]=nullFill((unique_Dict[asset]-mean_Dict[asset])*(unique_Dict[asset]-mean_Dict[asset]),0)}}
            // m=diag(unique_list);
            append!(param_list,pow(0.5,double(i)/double(D)));
            append!(covar_list,pow(0.5,double(i)/double(D))*m);
        }};
        EWMA_covar=sum(covar_list)/sum(param_list); //EWMA加权处理(特质收益率协方差矩阵)
    
        //Step2. Newey-West调整
        // 假设现在是调仓日
        unique_df=select symbol,date,unique_return from loadTable("{self.combine_database}","{self.individual_daily_table}") where date>=date({self.start_uniqueCov_ts}) and date<=date({self.end_uniqueCov_ts}) and class="OLS";
        unique_matrix=select unique_return from unique_df where symbol in code_list pivot by date,symbol;
        code_df=table(code_list as `symbol);
        total_ts_list=exec distinct(date) as date from unique_df;
        total_ts_list=total_ts_list.sort(); // 正序排序[1(上一个调仓周期开始时间),2,...,T-d](T为本次调仓周期开始时间)
        D=min(2,count(total_ts_list));   // 这里滞后期D需要设置为5,但设置成5算的太慢了,就设置成2
        total_period=min(15,count(total_ts_list));  // 这里不一定是15,只是Barra官方给的是15
    
        bartlett_list=[];  // (外循环)Bartlett权重
        COVAR_list1=[];  // (外循环)Bartlett权重加权
        COVAR_list2=[];
        for (d in seq(1,D)){{
            covar_list1=[]; // (内循环)u*transpose(u)构成的矩阵
            covar_list2=[]; // (内循环)transpose(u)*u构成的矩阵
            param_list=[]; // (内循环)参数权重
            for (t in seq(0,total_period-d-1)){{
                unique_table_t=sql(select=sqlCol(code_list),from=unique_matrix,where=[<date=total_ts_list[t]>]).eval(); // t期的特异性收益率矩阵
                unique_matrix_t=matrix(unique_table_t);
                unique_table_d=sql(select=sqlCol(code_list),from=unique_matrix,where=[<date=total_ts_list[t+d]>]).eval(); // t+d期的特异性收益率矩阵
                unique_matrix_d=matrix(unique_table_d);
                n=count(code_list);
                m1=matrix(DOUBLE,n,n);
                result_Dict=table(nullFill(abs(unique_matrix_d*unique_matrix_t),0))[0]; // 这里加了绝对值保证乘出来的是正数
                for (i in seq(0,n-1)){{
                    m1[i,i]=result_Dict[`col+string(i)];
                }};
                m2=transpose(m1);
                append!(param_list,pow(0.5,double(total_period-d-t)));
                append!(covar_list1,pow(0.5,double(total_period-d-t))*m1);
                append!(covar_list2,pow(0.5,double(total_period-d-t))*m2);
            }};
            append!(COVAR_list1,(1-double(d)/(double(D)+1))*sum(covar_list1)/sum(param_list));
            append!(COVAR_list2,(1-double(d)/(double(D)+1))*sum(covar_list2)/sum(param_list));
            append!(bartlett_list,(1-double(d)/(double(D)+1)))
            }};
            
        // 得到最终Newey-west调整后的因子收益率协方差矩阵
        posPeriod=int({self.t});// 当前调仓周期
        final_covar_table=posPeriod*(EWMA_covar+sum(COVAR_list1)+sum(COVAR_list2));
        final_covar_table=table(code_list as `symbol,diag(final_covar_table) as `unique_cov);
        final_covar_table[`date]={self.end_uniqueCov_ts};
        final_covar_table=select symbol,date,unique_cov from final_covar_table;
        
        // 添加数据至uniqueCov_table
        loadTable("{self.combine_database}","{self.uniqueCov_table}").append!(final_covar_table);
        """

    def BackTest(self):
        """多因子回测框架核心"""
        # 建立本地路径(资产收益率协方差矩阵保存路径)
        init_path(path_dir=self.Assetresult_pathdir)
        for benchmark in self.benchmark_list:
            init_path(path_dir=r"{}\{}".format(self.Assetresult_pathdir,str(benchmark)))

        # 若存在industry_list，则加入至factor_list的末尾
        self.factor_list+=self.industry_list

        # Step0.Init ResultTable
        self.init_ResultDataBase(dropDatabase=True)
        period_Dict=self.session.run(rf"""pt=select period,start_date as date from loadTable("{self.combine_database}","{self.template_table}"); dict(string(int(pt[`period])),pt[`date])""")
        for benchmark in tqdm.tqdm(self.benchmark_list,desc=f"Calculating result of RiskModel"):
            self.benchmark=benchmark
            self.session.run(self.summary_command())    # Step1. summary_result
            self.total_period_list=sorted([i for i in self.session.run(rf"""select distinct(int(period)) as period from loadTable("{self.result_database}","{self.summary_table}") where Benchmark="{self.benchmark}" """)['period'].tolist()])[1:] # 把第一个调仓日去掉,相当于第一个调仓区间是样本区间
            for period in tqdm.tqdm(self.total_period_list,desc=f"Calculating factorR_Cov+assetR_Cov on {self.benchmark}"):
                self.current_period=period
                self.current_date=pd.Timestamp(period_Dict[str(self.current_period)]).strftime('%Y.%m.%d')
                data=self.factorCov_func() # Step3.FactorReturn Cov Estimation+AssetReturn Cov Save
                data.to_feather(rf"{self.Assetresult_pathdir}\{self.benchmark}\{str(self.current_date).replace('.','')}.feather")   # 保存为feather格式

if __name__=="__main__":
    session=ddb.session()
    session.connect("localhost",8848,"admin","123456")
    pool=ddb.DBConnectionPool("localhost",8848,10,"admin","123456")
    from factor_func.Data_func import RiskModel_data as R
    F=Risk_Backtest(
        # 基础信息
        session=session,pool=pool,start_date="2020.01.04",end_date="2025.01.06",posPeriod=20,
        # 数据准备
        symbol_database="dfs://stock_cn/RiskModel",symbol_table="symbol",Symbol_prepareFunc=R.prepare_symbol_data,
        benchmark_database="dfs://stock_cn/RiskModel", benchmark_table="benchmark",benchmark_list=["b000001","b000985"],Benchmark_prepareFunc=R.prepare_benchmark_data,
        factor_database="dfs://stock_cn/Riskfactor",factor_table="factor",factor_list=["Quantum10","Quantum5","Volume10","Volume5"],
        Factor_prepareFunc=R.prepare_factor_data,
        Industry_prepareFunc=R.prepare_industry_data,industry_list=[
            "交通运输","传媒","公用事业","农林牧渔","医药生物","商贸零售","国防军工","基础化工",
            "家用电器","建筑材料","建筑装饰","房地产","有色金属","机械设备","汽车","煤炭","环保","电力设备",
            "电子","石油石化","社会服务","纺织服饰","综合","美容护理","计算机","轻工制造","通信","钢铁",
            "银行","非银金融","食品饮料"],
        combine_database="dfs://stock_cn/RiskModel",combine_table="combination",Combine_prepareFunc=R.prepare_combine_data,

        # 因子结果数据库
        Factorresult_database="dfs://stock_cn/RiskModel_result",     # 将AssetResult(资产收益率协方差矩阵)保存在本地
        Assetresult_pathdir=r"E:/苗欣奕的东西/行研宝/data/stock_cn/factor/assetR_cov/20250106"
    )

    # F.init_SymbolDatabase()
    # F.add_SymbolData()
    # F.init_BenchmarkDatabase()
    # F.add_BenchmarkData()
    # F.init_FactorDatabase(dropDatabase=False)
    # F.add_FactorData()
    # F.add_IndustryData()
    # # 如果原始数据没有变化，那么不用运行init_CombineDatabase()与add_CombineData()
    # F.init_CombineDataBase()    # 注：industry_factor要加在一定在factor后面
    #  F.add_CombineData()
    F.BackTest()
