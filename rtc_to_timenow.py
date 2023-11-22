import datetime
import pytz


def calculate_real_time(rtc_value,delay):
    # リアルタイムクロックの値から1601年1月1日を基準にした秒数を計算
    # 1秒 = 10^7 クロックサイクル
    seconds_since_1601 = rtc_value / 10**7

    # 1601年1月1日からの秒数をdatetimeオブジェクトに変換
    base_datetime = datetime.datetime(1601, 1, 1)
    actual_datetime = base_datetime + \
        datetime.timedelta(seconds=seconds_since_1601)

    # 日本時間に変換
    japan_timezone = pytz.timezone('Asia/Tokyo')
    actual_datetime_japan = actual_datetime.replace(
        tzinfo=pytz.utc).astimezone(japan_timezone)
    
        # delayの秒数を足す
    actual_datetime_japan += datetime.timedelta(seconds=delay)

    # 実際の時間を出力
    # print(actual_datetime_japan)
    return actual_datetime_japan
