import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from utils_gist import *
from sklearn import preprocessing
from CV3 import Run3InGIST

"""此时返回的是按照顺序排列的testing文件名"""
def load_test(path):
    file_test=[]
    for filename in os.listdir(path):              #listdir的参数是文件夹的路径
        if filename=='.DS_Store':     #防止读取mac文件下的.ds_store文件
            continue
        file_test.append(filename)
    file_test.sort(key=lambda x:int(x[:-4]))
    return file_test

file_test=load_test(r"./testing")

print(file_test)


"""准备读取test文件，并且进行测试"""
def load_test_image(file_test,dim,path):

    N=len(file_test)
    # set dimension of each picture
    d = 2048

    # Initialization
    tiny_image = np.zeros((N, d))    #有多少张图片就有多少个N
    #
    for i in range(len(file_test)):
        img = cv2.imread(os.path.join(path,'testing/'+file_test[i]), cv2.IMREAD_GRAYSCALE)
        f = Run3InGIST.Gist_features(img)
        tiny_image[i, :] = (f - np.mean(f)) / np.std(f)

    # scaler = preprocessing.StandardScaler().fit(tiny_image)
    # tiny_image = scaler.transform(tiny_image)

    return tiny_image
    # return np.array(2900,500)



"""生成txt文件"""
def create_txt(label_test,path,file_test):

    dict = {'bedroom': 1, 'coast': 2, 'forest': 3, 'highway': 4, 'industrial': 5, 'insidecity': 6,
            'kitchen': 7, 'livingroom': 8, 'mountain': 9, 'office': 10, 'openCountry': 11, 'store': 12,
            'street': 13, 'suburb': 14, 'tallBuilding': 15}

    results=[]
    f=open(path+'/test_result_knn.txt','w')
    for i in range(len(file_test)):
        results.append(file_test[i]+" "+list(dict.keys())[list(dict.values()).index(label_test[i])])
        f.write(file_test[i]+" "+list(dict.keys())[list(dict.values()).index(label_test[i])]+'\n')


#%%
#
path = os.path.abspath('.')


tiny_image, label = get_tiny_image(path,dim)


#%%
#
x_train, x_test, f_train, f_test = train_test_split(tiny_image, label, test_size=.05)

#%%

clf = KNeighborsClassifier(n_neighbors=1)
clf.fit(x_train, f_train)
# predictions = clf.predict(x_test, f_test)
clf.score(x_test,f_test)

print("score:",clf.score(x_test,f_test))

"""处理测试集图片"""
tiny_image_test=load_test_image(file_test,dim,path)

"""生成测试集label"""
label_test=clf.predict(tiny_image_test)

"""生成测试集txt"""
create_txt(label_test,path,file_test)
print('--------------------')
