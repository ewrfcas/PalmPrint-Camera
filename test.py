import cv2
import numpy as np
import Hourglass as HG
from skimage import io,color,transform
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
import pandas as pd

#大约1cm=18.85
cm_std=18.85

static_result=pd.read_csv('static_results.csv',encoding='gbk')
model=HG.model(input_shape=(256,256,1),labels=20,nstack=6,level=4,filters=256)
model.load_weights('CNN_Hourglass256c.h5')


# 计算各种距离长度
def cal_length(life_points, int_points, aff_points, finger_points):
    life_length = euclidean(life_points[0], life_points[1]) + euclidean(life_points[1], life_points[2]) + euclidean(
        life_points[2], life_points[3]) + euclidean(life_points[3], life_points[4])
    int_length = euclidean(int_points[0], int_points[1]) + euclidean(int_points[1], int_points[2]) + euclidean(
        int_points[2], int_points[3]) + euclidean(int_points[3], int_points[4])
    aff_length = euclidean(aff_points[0], aff_points[1]) + euclidean(aff_points[1], aff_points[2]) + euclidean(
        aff_points[2], aff_points[3]) + euclidean(aff_points[3], aff_points[4])
    hand_length = euclidean(finger_points[3], finger_points[4])
    finger_width = (euclidean(finger_points[0], finger_points[1]) + euclidean(finger_points[1], finger_points[2])) / 2

    return life_length, int_length, aff_length, hand_length, finger_width

# 计算角度（判断是否过线;感情线上翘,下翘）
def cal_angle(life_points, int_points, aff_points, finger_points, finger_width):
    # 判断生命线是否过中线True,False 中线方程k=[y2－y1]/[x2－x1]
    x1 = finger_points[3][0]
    x2 = finger_points[4][0]
    y1 = finger_points[3][1]
    y2 = finger_points[4][1]
    k = (y2 - y1) / (x2 - x1 + 1e-6)
    b = y1 - k * x1
    flag_list = []
    life_over = False
    for lp in life_points:
        d = k * lp[0] + b - lp[1]
        if d >= 0:
            flag_list.append(1)
        else:
            flag_list.append(-1)
        if len(flag_list) >= 2 and flag_list[-1] != flag_list[-2]:
            life_over = True
            break

    # 智慧线是否超过小指和无名指之间True,False，停留于小指和无名指之间距离int_dis=d[-1]/finger_width
    flag_list = []
    int_over = False
    b = finger_points[2][1] - k * finger_points[2][0]
    for ip in int_points:
        d = k * ip[0] + b - ip[1]
        if d >= 0:
            flag_list.append(1)
        else:
            flag_list.append(-1)
        if len(flag_list) >= 2 and flag_list[-1] != flag_list[-2]:
            int_over = True
            break
    int_dis = abs(k * int_points[-1][0] + b - int_points[-1][1])
    if not int_over:
        int_dis *= (-1)
    int_dis /= finger_width

    # 感情线终点位置，是否过中指到食指之间True,False
    flag_list = []
    aff_over = False
    b = finger_points[0][1] - k * finger_points[0][0]
    for ap in aff_points:
        d = k * ap[0] + b - ap[1]
        if d >= 0:
            flag_list.append(1)
        else:
            flag_list.append(-1)
        if len(flag_list) >= 2 and flag_list[-1] != flag_list[-2]:
            aff_over = True
            break
    aff_dis = abs(k * aff_points[-1][0] + b - aff_points[-1][1])
    if not aff_over:
        aff_dis *= (-1)
    aff_dis /= finger_width
    # 计算感情线最后一个点到倒数第三个点连线与中线的夹角 cosθ=(vec1*vec2)/(|vec1|*|vec2|)
    vec1 = [finger_points[4][0] - finger_points[3][0], finger_points[4][1] - finger_points[3][1]]
    vec2 = [aff_points[-1][0] - aff_points[-3][0], aff_points[-1][1] - aff_points[-3][1]]
    cosxita = (np.dot(vec1, vec2)) / (np.sqrt(vec1[0] ** 2 + vec1[1] ** 2) * np.sqrt(vec2[0] ** 2 + vec2[1] ** 2))
    aff_curve = np.arccos(cosxita) / np.pi * 180. -90.

    return life_over, int_over, int_dis, aff_over, aff_dis, aff_curve


# 计算深浅
def cal_deep(life_points, int_points, aff_points, img, deep_range=1):
    # 深浅
    life_deep = 0
    for lp in life_points:
        life_deep += (
            np.mean(img[max(0, int(lp[0] - deep_range)):int(lp[0] + deep_range + 1),
                    max(0, int(lp[1] - deep_range)):int(lp[1] + deep_range + 1)]))
    life_deep /= len(life_points)
    life_deep = 255. - life_deep

    int_deep = 0
    for ip in int_points:
        int_deep += (
            np.mean(img[max(0, int(ip[0] - deep_range)):int(ip[0] + deep_range + 1),
                    max(0, int(ip[1] - deep_range)):int(ip[1] + deep_range + 1)]))
    int_deep /= len(int_points)
    int_deep = 255. - int_deep

    aff_deep = 0
    for ap in aff_points:
        aff_deep += (
            np.mean(img[max(0, int(ap[0] - deep_range)):int(ap[0] + deep_range + 1),
                    max(0, int(ap[1] - deep_range)):int(ap[1] + deep_range + 1)]))
    aff_deep /= len(aff_points)
    aff_deep = 255. - aff_deep

    return life_deep, int_deep, aff_deep

# 算命函数
def fortune_telling(life_length_per,life_over,int_length_per,int_over,int_dis_per,aff_length_per,aff_over,aff_dis_per,aff_curve_per):
    if life_length_per>66.66:
        life_length_des='生命力强，对疾病的抵抗力强，不容易生病'
    elif life_length_per>20. and life_length_per<=66.66:
        life_length_des='生命强度适中，抵抗力良好'
    else:
        life_length_des='生命力较为衰弱，容易劳累'

    if life_over:
        life_over_des='积极主动'
    else:
        life_over_des='被动消极'

    if int_length_per>66.66:
        int_length_des='非常精明'
    elif int_length_per>20. and int_length_per<=66.66:
        int_length_des='智力适中'
    else:
        int_length_des='大智若愚'

    if int_over:
        int_over_des='精明过度'
    else:
        int_over_des='无过度的精明'

    if int_dis_per>66.66:
        int_dis_des='精明过度'
    elif int_dis_per>33.33 and int_dis_per<=66.66:
        int_dis_des='最合适，智慧而不过分外露'
    else:
        int_dis_des='老实'

    if aff_length_per>66.66:
        aff_length_des='情深义重'
    elif aff_length_per>20. and aff_length_per<=66.66:
        aff_length_des='感情中庸'
    else:
        aff_length_des='淡薄情义'

    if aff_over:
        aff_over_des='重情义之人'
    else:
        aff_over_des='更注重人的外表'

    if aff_dis_per>66.66:
        aff_dis_des='注重精神之爱'
    elif aff_dis_per>20. and aff_dis_per<=66.66:
        aff_dis_des='对于爱介于精神与肉体之间'
    else:
        aff_dis_des='属肉体之爱，不注重山盟海誓'

    if aff_curve_per<30.:
        aff_curve_des='对感情不择手段，任性，牺牲一切'
    else:
        aff_curve_des='对感情坦然，容易接受'

    return life_length_des,life_over_des,int_length_des,int_over_des,int_dis_des,aff_length_des,aff_over_des,aff_dis_des,aff_curve_des


weight=1200
height=700
cap = cv2.VideoCapture(1)
while True:
    ret,frame = cap.read()
    frame=transform.resize(frame,(height,weight),mode='constant')
    frame2 = frame.copy()
    frame=color.rgb2gray(frame)
    pt1=(int(weight/4),int(height/6))
    pt2=(int(weight*3/4),int(height*5/6))
    cv2.rectangle(frame, pt1, pt2, (255, 255, 255), 3)
    cv2.imshow('PalmPrint',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        img=frame2[pt1[1]:pt2[1],pt1[0]:pt2[0]]
        img=color.rgb2gray(img)
        img=transform.resize(img,(256,256),mode='constant')
        io.imsave('img.jpg',img)

        img = io.imread('img.jpg')
        prediction = model.predict(img.reshape(1, 256, 256, 1))[-1][0]
        x = []
        y = []
        for i in range(prediction.shape[2]):
            xt = int((np.argmax(prediction[:, :, i]) % 64 / 64) * 256)
            yt = int(int(np.argmax(prediction[:, :, i]) / 64) / 64 * 256)
            x.append(xt)
            y.append(yt)

        life_x = x[0:5]
        life_y = y[0:5]
        int_x = x[5:10]
        int_y = y[5:10]
        aff_x = x[10:15]
        aff_y = y[10:15]
        finger_x = x[15:20]
        finger_y = y[15:20]
        life_points = []
        int_points = []
        aff_points = []
        finger_points = []
        for i in range(5):
            life_points.append([life_x[i], life_y[i]])
            int_points.append([int_x[i], int_y[i]])
            aff_points.append([aff_x[i], aff_y[i]])
            finger_points.append([finger_x[i], finger_y[i]])

        life_length, int_length, aff_length, hand_length, finger_width = cal_length(life_points, int_points, aff_points, finger_points)
        life_over, int_over, int_dis, aff_over, aff_dis, aff_curve = cal_angle(life_points, int_points, aff_points, finger_points, finger_width)
        life_deep, int_deep, aff_deep = cal_deep(life_points, int_points, aff_points, img)

        columns = ['指标', '数值', '=%', '>%', '描述']

        life_length_per = round(
            len(np.where(static_result['生命线指宽'].values < (life_length / finger_width))[0]) / 5502 * 100, 2)
        life_over_per = round(len(np.where(static_result['生命线是否过中线'].values == life_over)[0]) / 5502 * 100, 2)
        int_length_per = round(
            len(np.where(static_result['智慧线指宽'].values < (int_length / finger_width))[0]) / 5502 * 100, 2)
        int_over_per = round(len(np.where(static_result['智慧线是否过小指无名指中点'].values == int_over)[0]) / 5502 * 100, 2)
        int_dis_per = round(len(np.where(static_result['智慧线到中点距离'].values < (int_dis))[0]) / 5502 * 100, 2)
        aff_length_per = round(
            len(np.where(static_result['感情线指宽'].values < (aff_length / finger_width))[0]) / 5502 * 100, 2)
        aff_over_per = round(len(np.where(static_result['感情线是否过食指中指中点'].values == aff_over)[0]) / 5502 * 100, 2)
        aff_dis_per = round(len(np.where(static_result['感情线到中点距离'].values < (aff_dis))[0]) / 5502 * 100, 2)
        aff_curve_per = round(len(np.where(static_result['感情线末尾与中线夹角'].values < (aff_curve))[0]) / 5502 * 100, 2)

        life_length_des, life_over_des, int_length_des, int_over_des, int_dis_des, aff_length_des, aff_over_des, aff_dis_des, aff_curve_des = fortune_telling(
            life_length_per, life_over, int_length_per, int_over, int_dis_per, aff_length_per, aff_over, aff_dis_per,
            aff_curve_per)

        data_all = []
        data_all.append(
            ['生命线长度', str(round(life_length / cm_std, 2)) + 'cm', '/', str(life_length_per) + '%', life_length_des])
        data_all.append(['生命线是否过中点', str(life_over), str(life_over_per) + '%', '/', life_over_des])
        data_all.append(['生命线深度', str(round(life_deep, 2)), '/', '/', '深:身体好,浅:多病'])
        data_all.append(
            ['智慧线长度', str(round(int_length / cm_std, 2)) + 'cm', '/', str(int_length_per) + '%', int_length_des])
        data_all.append(['智慧线是否过小指无名指中点', str(int_over), str(int_over_per) + '%', '/', int_over_des])
        data_all.append(['智慧线到中点距离', str(round(int_dis, 4)), '/', str(int_dis_per) + '%', int_dis_des])
        data_all.append(['智慧线深度', str(round(int_deep, 2)), '/', '/', '深:偏智力,浅:偏体力'])
        data_all.append(
            ['感情线长度', str(round(aff_length / cm_std)) + 'cm', '/', str(aff_length_per) + '%', aff_length_des])
        data_all.append(['感情线是否过食指中指中点', str(aff_over), str(aff_over_per) + '%', '/', aff_over_des])
        data_all.append(['感情线到中点距离', str(round(aff_dis, 4)), '/', str(aff_dis_per) + '%', aff_dis_des])
        data_all.append(['感情线末尾与中线夹角', str(round(aff_curve, 2)) + '˚', '/', str(aff_curve_per) + '%', aff_curve_des])
        data_all.append(['感情线深度', str(round(aff_deep, 2)), '/', '/', '深:感情细腻,浅:豪放'])
        df = pd.DataFrame(data_all, columns=columns)
        df.to_excel('result.xls', index=None, encoding='gb2312')

        # print('生命线:', str(life_length/cm_std)+'cm', str(life_length/finger_width)+'倍指宽'+'(平均:3.15)', '深度:',life_deep)
        # if life_over:
        #     print('生命线超过命运线')
        # else:
        #     print('生命线未超过命运线')
        # print('----------------------------------------------------')
        # print('智慧线:', str(int_length/cm_std)+'cm', str(int_length/finger_width)+'倍指宽'+'(平均:2.85)', '深度:', int_deep)
        # if int_over:
        #     print('智慧线超过小指和无名指中点')
        # else:
        #     print('智慧线未超过小指和无名指中点')
        # print('智慧线到小指和无名指中点距离:',int_dis)
        # print('----------------------------------------------------')
        # print('感情线:', str(aff_length/cm_std)+'cm', str(aff_length/finger_width)+'倍指宽'+'(平均:2.71)', '深度:', aff_deep)
        # if aff_over:
        #     print('感情线超过中指和食指中点，重情义O(∩_∩)O')
        # else:
        #     print('感情线未超过中指和食指中点，重肉体⁄(⁄ ⁄•⁄ω⁄•⁄ ⁄)⁄')
        # print('感情线到小指和无名指中点距离:',aff_dis)
        # print('感情线与中线夹角:',aff_curve,'˚')
        # print('----------------------------------------------------')
        # print('手掌长度:', str(hand_length/cm_std)+'cm')
        # print('指宽:', str(finger_width/cm_std)+'cm')

        plt.imshow(img, cmap='gray')
        plt.scatter(life_x, life_y, c='g', s=10)
        plt.scatter(int_x, int_y, c='b', s=10)
        plt.scatter(aff_x, aff_y, c='r', s=10)
        plt.scatter(finger_x, finger_y, c='cyan', s=10)
        plt.savefig('img_labeled.jpg')
        plt.show()
# cap.release()
# cv2.destroyAllWindows()


