from scipy import stats
import numpy as np
import matplotlib.pyplot as plt


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

    return p_value < alpha

def QQplot(data):
    # 正規QQプロットの作成
    probplot_data = stats.probplot(data, plot=plt)

    # グリッド線の追加（オプション）
    plt.grid(True, which='both', linestyle='--', color='gray', alpha=0.5)
    # グラフの表示
    plt.title('Normal QQ Plot')
    plt.show()
    
def paired_t_test(alpha,data1,data2):
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
    return p_value < alpha

