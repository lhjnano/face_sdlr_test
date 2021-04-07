import simple_face

"""
!!! Python v3.6 에서 라이브러리 설치 가능 
"""

image = simple_face.load_image('test.jpg')
locations = simple_face.face_locations(image)
print('{}'.format(locations))

landmark = simple_face.face_landmark(image, locations)
print('{}'.format(landmark))

registerList = simple_face.face_registerList('simple_face')

isExist = simple_face.face_find(registerList, 'biden')

if isExist is False:
	if simple_face.face_register('simple_face', 'simple_face.clf', image, 'biden') is False:
		print('face_register is Fail')
		exit()
	
image2 = simple_face.load_image('test2.jpg')
subimage = image2[20:120, 200:300]
simple_face.face_super_resolution(subimage, 'sr.jpg')

locations2 = simple_face.face_locations(image2)
print('{}'.format(locations2))
simple_face.face_super_resolution_prediction('simple_face.clf', image2, locations2, 'save.jpg')
