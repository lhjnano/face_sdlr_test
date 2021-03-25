from face_sdlr import face
import cv2
import os

"""
!!! Python v3.6 에서 라이브러리 설치 가능 
"""

""" 
Example 1  : Face Locations

Result :
[(288, 67, 490, 380)]
"""

image = face.load_image('test.jpg')
locations = face.face_locations(image)

print('{}'.format(locations))

"""
Example 2 : Face Landmark

Result :
[array([-0.03791746,  0.19169572,  0.06861898, -0.0282283 , -0.08486373,
        0.08860959, -0.04261846, -0.09161349,  0.06719191, -0.0022766 ,
        0.18790439, -0.00286531, -0.29673395, -0.01347055,  0.00674275,
        0.1482823 , -0.09823523, -0.07230916, -0.21035136, -0.05761878,
       -0.00909248,  0.04019308,  0.07679158, -0.06670092, -0.10927769,
       -0.2494079 , -0.08584137, -0.08366247, -0.05286765, -0.13397746,
        0.07220051, -0.04105808, -0.22543755, -0.0267313 , -0.06155687,
       -0.03695586, -0.03605538, -0.0802984 ,  0.12767781,  0.02915895,
       -0.14389339,  0.10178754,  0.02992794,  0.18073209,  0.27124834,
        0.02137555,  0.04222105, -0.11115775,  0.07836883, -0.21191591,
        0.01706373,  0.0934803 ,  0.18663412,  0.08293638,  0.01117594,
       -0.04044866,  0.04963051,  0.17058556, -0.23337956,  0.07026219,
        0.03337226, -0.03891805, -0.02473803, -0.00631615,  0.1383325 ,
        0.14785577, -0.08169433, -0.14744395,  0.16759826, -0.15120079,
       -0.11713625,  0.04753211, -0.10203398, -0.18268518, -0.34993586,
        0.00508806,  0.27967942,  0.08600482, -0.27546677, -0.08409377,
       -0.06804155,  0.00063778, -0.02898233,  0.06715952, -0.08079108,
       -0.14045084, -0.07807717, -0.00296692,  0.29327819, -0.09606205,
       -0.04175656,  0.26967326,  0.05590236, -0.09095426, -0.01018396,
        0.1216098 , -0.04951667, -0.03109003, -0.09025122, -0.01686882,
       -0.02808287, -0.11221066, -0.06740697,  0.11187165, -0.19424039,
        0.18658137, -0.03377245, -0.04651824, -0.03540339, -0.05577226,
       -0.01345799, -0.00452955,  0.21725067, -0.20066985,  0.1963103 ,
        0.24589233, -0.09225892,  0.0362424 ,  0.0190527 ,  0.14373024,
       -0.02496282,  0.02512344, -0.16025437, -0.15913904,  0.02536015,
       -0.01429408, -0.01731917,  0.07549746])]
"""

landmark = face.landmark(image, locations)
print('{}'.format(landmark))


""" 
Example 3 : Register List 

Result : 
['D:\\sr_test\\simple_face\\biden\\17a8dfdf.jpg']
"""

registerList = face.registerList('simple_face')
for registerPath in registerList :
    print(registerPath)

"""
Example 4 : Register faces 

이미지 파일이 훈련 모델에 맞게 저장됩니다. 
재 호출하면 같은 파일이 두번 저장됩니다. 
""" 
hasBiden = False
for registerPath in registerList :
    if registerPath.find('biden'):
        hasBiden = True
        break
        
if hasBiden is False:
    face.register(model_name='simple_face', image=image, label='biden')



"""
Example 5 : Train face 

Result :
모델 파일 생성 : simple_face.clf
"""
face.train_face('simple_face.clf')


"""
Example 6 : Super Resolution

Result :
얼굴 확대 파일 생성 : sr.jpg
"""
if os.path.isfile('simple_face.clf') == False: 
    exit()
image = face.load_image('test2.jpg')

srImage = face.super_resolution(image[20:120, 200:300])
cv2.imwrite('sr.jpg', srImage)

"""
Example 7 : Predict face 

Result :
얼굴 예측 파일 생성 : save.jpg
"""
locations = face.face_locations(image)
who = face.predict_faces('simple_face.clf', image, locations)
image = face.show_prediction(image, who)
cv2.imwrite('save.jpg', image)
