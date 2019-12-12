#%%
import numpy as np
import cv2
import os
from sklearn import preprocessing
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

def cv_show(name, image):
    cv2.imshow(name, image)
    cv2.waitKey()
    cv2.destroyAllWindows()




def gen_sift_features(gray, step_size):
    gray = cv2.pyrDown(gray)
    print(gray.shape)
    dense = cv2.xfeatures2d.SIFT_create(103)
    kp = [cv2.KeyPoint(x, y, step_size) for y in range(0, gray.shape[0], step_size) for x in range(0, gray.shape[1], step_size)]
    kp, desc = dense.compute(gray, kp)
    return kp, desc




def get_dense_sampling_images(path, dim=8):
    # Number of each class (each class contains 100 pictures)
    N = 100

    # Classes dictionary
    Classes = ['bedroom', 'Coast', 'Forest', 'Highway', 'industrial', 'Insidecity', 'kitchen',
               'livingroom', 'Mountain', 'Office', 'OpenCountry', 'store', 'Street', 'Suburb', 'TallBuilding']
    dict = {'bedroom': 1, 'Coast': 2, 'Forest': 3, 'Highway': 4, 'industrial': 5, 'Insidecity': 6,
            'kitchen': 7, 'livingroom': 8, 'Mountain': 9, 'Office': 10, 'OpenCountry': 11, 'store': 12,
            'Street': 13, 'Suburb': 14, 'TallBuilding': 15}

    # set dimension of each picture
    d = dim ** 2

    # Initialization
    dense_sampling_image_total = np.zeros((1,128))
    dense_sampling_images = []
    #
    label = []
    for Class in Classes:
        for i in range(N):
            img = cv2.imread(os.path.join(path,'training', Class, str(i)+'.jpg'), cv2.IMREAD_GRAYSCALE)
            kp, desc = gen_sift_features(img, 4)                                  #每张图片的数组 e.g. 2000*128
            dense_sampling_image_total = np.vstack((dense_sampling_image_total,desc))
            dense_sampling_images.append(desc)
            label.append(dict[Class])

    return dense_sampling_image_total[1:,:], dense_sampling_images, label
    
#%%
path = os.path.abspath('.')
dense_sampling_image_total, dense_sampling_images, label = get_dense_sampling_images(path)
print(1)
#%%
import numpy as np
dense_sampling_images = np.array(dense_sampling_images)


#%%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(dense_sampling_images, label, test_size=.15)


#%%
print(X_train[0].shape)
X = X_train[0]
for i in range(1,len(X_train)):
    X = np.vstack((X,X_train[i]))
print(X.shape)

#%%
from sklearn.cluster import MiniBatchKMeans,KMeans

cluster = MiniBatchKMeans(n_clusters=500, batch_size=50)
# cluster = KMeans(n_clusters=500,random_state= 54, max_iter= 3000,n_jobs = 2,n_init = 2)
cluster.fit(X)
from sklearn.externals import joblib

#%%
km_model = cluster

#%%
from sklearn import preprocessing
def get_image_presentation(dense_sampling_images, centroids_model):
    image_presentation = np.zeros((len(dense_sampling_images), len(centroids_model.cluster_centers_)))
    histograms = []
    count = 0
    for image in dense_sampling_images:
        hist = centroids_model.predict(image)
        histograms.append(hist)
        for label in hist:
            image_presentation[count][label] += 1
        image_presentation[count] /= len(image)
        count += 1

    return image_presentation, histograms

image_presentation_train, histograms_train = get_image_presentation(X_train, km_model)
image_presentation_test, histograms_test = get_image_presentation(X_test, km_model)
std=preprocessing.StandardScaler()
image_presentation_train=std.fit_transform(image_presentation_train)
image_presentation_test=std.fit_transform(image_presentation_test)
# np.save('image_presentation3.npy', image_presentation)
#%%
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
# clf = OneVsRestClassifier(LogisticRegression()).fit(image_presentation_train, y_train)
clf = OneVsRestClassifier(SVC()).fit(image_presentation_train, y_train)
# clf = GaussianNB().fit(image_presentation_train,y_train)
#%%
clf.score(image_presentation_test, y_test)
#%%
print(X_train.shape)
print(clf.score(image_presentation_test, y_test))


# # %%
# from sklearn.externals import joblib
# from sklearn.cluster import MiniBatchKMeans,KMeans
# path = os.path.abspath('.')
# SiftImage=load_test(r"./testing")
# TestingSiftImageSave ,TestingSiftImage = load_test_image(SiftImage,path)
# print(2)
#
# GetSiftImage , SiftHistoram = get_image_presentation(TestingSiftImage,km_model)
# print(3)
# std=preprocessing.StandardScaler()
# GetSiftImage = std.fit_transform(GetSiftImage)
# label_test=clf.predict(GetSiftImage)
# print(4)
# create_txt(label_test,path,SiftImage)



