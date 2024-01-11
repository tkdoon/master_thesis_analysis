from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.stats.anova as anova
import pandas as pd
from itertools import combinations

def shapiro_wilk(alpha,data:list):
    """_summary_
    正規性が認められればTrue，認められなければFalseを返す．
    Args:
        alpha (_type_): _description_
        data (list): _description_

    Returns:
        _type_: _description_
    """
    # シャピロ-ウィルク検定の実行
    statistic, p_value = stats.shapiro(data)
    return p_value < alpha,p_value

def QQplot(data):
    # 正規QQプロットの作成
    probplot_data = stats.probplot(data, plot=plt)

    # グリッド線の追加（オプション）
    plt.grid(True, which='both', linestyle='--', color='gray', alpha=0.5)
    # グラフの表示
    plt.title('Normal QQ Plot')
    plt.show()
    
def paired_t_test(data1,data2,alpha=0.05):
    """_summary_
    統計的な有意差が認められればTrue，認められなければFalseを返す．
    Args:
        alpha (_type_): _description_
        data1 (_type_): _description_
        data2 (_type_): _description_

    Returns:
        _type_: _description_
    """
    t_statistic,p_value=stats.ttest_rel(data1, data2)
    return p_value < alpha,p_value

def wilcoxon(data1,data2,alpha=0.05):
    """_summary_
    統計的な有意差が認められればTrue，認められなければFalseを返す．
    Args:
        alpha (_type_): _description_
        data1 (_type_): _description_
        data2 (_type_): _description_

    Returns:
        _type_: _description_
    """
    t_statistic,p_value=stats.wilcoxon(data1, data2,alternative="two-sided")
    return p_value < alpha,p_value

def repeated_anova(alpha,*data_list):
    subjects=["sub"+str(i) for i,data in enumerate(data_list)] 
    combined_list=[]
    for data in data_list:
        combined_list+=data
    points=np.array(combined_list)
    conditions=np.repeat(["con"+str(i) for i,data in enumerate(data_list)],len(data_list[0]))
    df = pd.DataFrame({'Point':points,'Conditions':conditions,'Subjects':subjects})
            
    aov=anova.AnovaRM(df)
    aov.fit()
    print(aov)
    p_value=aov.anova_table["Pr > F"]
    return p_value < alpha,p_value

def friedman(alpha,*data_list):
    t_statistic,p_value=stats.friedmanchisquare(*data_list)
    return p_value < alpha,p_value
    
def kruskal_wolis(alpha,*data_list):
    """_summary_
    統計的な有意差が認められればTrue，認められなければFalseを返す．
    Args:
        alpha (_type_): _description_

    Returns:
        _type_: _description_
    """
    t_statistic,p_value=stats.kruskal(*data_list)
    return p_value < alpha,p_value    
    

analyze_list=[
    
]

compare_list=[
    [0,12],[1,13],[2,14],[3,6,9,12],[4,7,10,13],[5,8,11,14],[15,16,17],[18,19,20],[21,22,23],[24,25,26],[27,28,29],[30,31,32],[33,34,35],[36,37,38],
]

def main():
    df_list=[]
    for i in range(3,15):
        read_path=fr"C:\Users\tyasu\Desktop\修士研究用\解析データ\parameters{i}.csv"
        df=read_df(read_path)
        df_list.append(df)
    print(3)
    
    for analyze_item in analyze_list:
        data_list=[]
        for df1 in df_list:
            data=df1.loc[analyze_item].values
            data_list.append(data)
        # np.array(data_list)
        res=[]
        for compare_item in compare_list:
            prepared_list=[]
            for num in compare_item:
                temp_list=[]
                for subject_num in  range(23):
                    temp_list.append(data_list[subject_num][num])
                prepared_list.append(temp_list)
            res.append(statistical_analysis(*prepared_list))
            
        write_path=fr"C:\Users\tyasu\Desktop\修士研究用\解析データ\{analyze_item}.csv"    
        write_csv(write_path,*res)
        
        
def holm_method(alpha=0.05,*p_values):
    sorted_p_list = sorted(p_values)
    indices_in_sorted_list = [sorted_p_list.index(value) for value in p_values]
    fixed_p_list=[p*(len(p_values)-i) for i,p in enumerate(sorted_p_list)]
    difference_list=[fixed_p<0.05 for fixed_p in fixed_p_list]
    return difference_list,fixed_p_list,indices_in_sorted_list
               
                    
def statistical_analysis(*data_list):
    for data in data_list:
        normality,normality_p=shapiro_wilk(data)
        print("シャピロウィルク","p値：",normality_p,"正規性あり" if normality else "正規性なし")
        QQplot(data)
        judge=input()
    
    if judge=="n":
       normality=False
    elif judge=="y":
        normality=True
    
    if(len(data_list)==2 and normality):
        difference,p_value=paired_t_test(0.05, data_list[0],data_list[1])
        print("対応のあるｔ検定","有意差あり：" if difference else "有意差なし","p値：",p_value)
        return p_value,difference
    elif(len(data_list)==2 and not normality):
        difference,p_value=wilcoxon(0.05, data_list[0], data_list[1])
        print("wilcoxonの符号順位検定","有意差あり：" if difference else "有意差なし","p値：",p_value)
        return p_value,difference
    elif(len(data_list)>=3 and normality):
        difference,p_value=repeated_anova(0.05,*data_list)
        if (difference):
            print("反復測定ANOVA","有意差あり：","p値：",p_value)
            combinations_list = list(combinations(data_list, 2))
            p_list=[]
            for combi in combinations_list:
                each_difference,each_p_value=paired_t_test(combi[0],combi[1])
                p_list.append(each_p_value)
            difference_list,fixed_p_list,indices_in_sorted_list=holm_method(0.05,*p_list)  
            res=[[combinations_list[i],difference_list[indice],fixed_p_list[indice]] for i,indice in enumerate(indices_in_sorted_list)]
            print("対応のあるｔ検定（多重比較）",res)
            return p_value,res
        else:
            print("反復測定ANOVA","有意差なし","p値：",p_value)
            return p_value,False
    elif(len(data_list)>=3 and not normality):
        difference,p_value=friedman(0.05, *data_list)    
        if (difference):
            print("friedman","有意差あり：","p値：",p_value)
            combinations_list = list(combinations(data_list, 2))
            p_list=[]
            for combi in combinations_list:
                each_difference,each_p_value=wilcoxon(combi[0],combi[1])
                p_list.append(each_p_value)
            difference_list,fixed_p_list,indices_in_sorted_list=holm_method(0.05,*p_list)  
            res=[[combinations_list[i],difference_list[indice],fixed_p_list[indice]] for i,indice in enumerate(indices_in_sorted_list)]
            print("wilcoxon（多重比較）",res)
            return p_value,res
        else:
            print("friedman","有意差なし：","p値：",p_value)
            return p_value,False
    else:
        return 
        

    

    
    
def read_df(path):
    df = pd.read_csv(path,header=1 )
    return df

def write_csv(path,*args):
    """可変長引数の中身はkey,value1,value2,...の形で渡してください"""
    import csv
    with open(path,'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        # データをCSVファイルに書き込む
        csv_writer.writerows(args)


if __name__ == '__main__':
    main()