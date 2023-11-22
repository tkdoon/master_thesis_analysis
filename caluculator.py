print("バスが旋回の場合→1,信号が青から赤に変わるとき→2")
which = int(input())
print("gを何倍した値をバスの加速度にしますか？")
a_multiplying_g = float(input())
speed_kmh = 40
g = 9.8

if(which == 1):
    print("gを何倍した値をバスの向心加速度にしますか？")
    circle_a_multiplying_g = float(input())
    print("回転半径は何メートルにしますか？")
    radius = float(input())

    a = g*a_multiplying_g
    circle_a = g*circle_a_multiplying_g
    speed_ms = speed_kmh*1000/60/60
    print(
        f"速さ：{speed_kmh}km/h = {speed_ms}m/s \n加速度：{a}m/s^2 \n向心加速度：{circle_a}m/s^2")

    distance_to_stop = speed_ms**2/2/a
    time_to_stop = speed_ms/a
    print(f"停止するまでに走る距離：{distance_to_stop}m \n停止するまでにかかる時間：{time_to_stop}s")
    circle_speed = (circle_a*radius)**0.5
    circle_speed_kmh = circle_speed/1000*60*60
    print(f"旋回速度：{circle_speed}m/s = {circle_speed_kmh}km/h")

    time_to_circle_speed = -(circle_speed-speed_ms)/a
    distance_to_circle_speed = (circle_speed**2-speed_ms**2)/(-2*a)

    print(
        f"旋回速度に達するまでに走る距離：{distance_to_circle_speed}m \n旋回速度に達するまでにかかる時間：{time_to_circle_speed}s")
elif(which == 2):
    blink_time = 4
    pedestrian_red_time = 2
    a = g*a_multiplying_g
    speed_ms = speed_kmh*1000/60/60
    print(f"速さ：{speed_kmh}km/h = {speed_ms}m/s \n加速度：{a}m/s^2")

    distance_to_stop = speed_ms**2/2/a
    time_to_stop = speed_ms/a
    print(f"停止するまでに走る距離：{distance_to_stop}m \n停止するまでにかかる時間：{time_to_stop}s")

    # そのままのスピードだと信号が黄色に変わるときに信号に到達するとする
    time_to_yellow_light = distance_to_stop/speed_ms
    distance_of_to_blink = distance_to_stop + \
        (blink_time+pedestrian_red_time-time_to_yellow_light)*speed_ms
    distance_turn_to_yellow_light = distance_to_stop - \
        (speed_ms*time_to_yellow_light-0.5*a*(time_to_yellow_light ** 2))
    print(
        f"バスが減速ポイントに到達したときに黄色信号になるまでの残り時間：{time_to_yellow_light}s \n点滅信号に変えるポイントと停止位置の距離：{distance_of_to_blink}m \n黄色信号に変わるときのバスと停止線の距離：{distance_turn_to_yellow_light}m")
