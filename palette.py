import numpy as np
import cv2
from sklearn.cluster import KMeans


input_name ="E:/Downloads/1.jpg"
fond=cv2.imread("E:/Downloads/fond.jpg")
img=cv2.imread(input_name)

def palette(colors):
    step=20
    hauteur=200
    largeur=100

    cv2.rectangle(fond, (1250, 1210), (1250+largeur,1210+hauteur), colors[0], -1)
    xtop = 1350+step
    xend = 1350+step+largeur
    for i, couleur in enumerate(colors[1:]):
        cv2.rectangle(fond, (xtop, 1210), ( xend, 1210+hauteur), colors[i], -1)
        xtop += largeur + step
        xend = xtop + largeur
    return fond

if img.shape[0]!=1080: 
    r=1080/img.shape[0]
    img=cv2.resize(img, ((int(img.shape[1]*r)), 1080), cv2.INTER_AREA)

clt = KMeans(n_clusters=5, random_state=1)
clt.fit(img.reshape(-1, 3))

colors_x = clt.cluster_centers_


x_offset=int((2000/2)-(img.shape[1]/2))
y_offset=40
fond[y_offset:y_offset+img.shape[0], x_offset:x_offset+img.shape[1]] = img

cv2.imwrite("E:/Downloads/test.jpg", palette(colors_x))