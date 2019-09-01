import cv2
import numpy as np

classes={'road':0,'sidewalk':1,'non-drivable':2,'person':3,'animal':4,'rider':5,'motorcycle':6,'bicycle':7,'rickshaw':8,'car':9,
        'truck':10,'bus':11,'train':12,'curb':13,'wall':14,'fence':15,'billboard':16,'traffic sign':17,'traffic light':18,
         'pole':19,'poles':20,'fallback':21,
        'building':22,'bridge':23,'vegetation':24,'sky':25,'background':26,'unlabelled':27}

def read_img(path):
    img=cv2.imread(path)
    cv2.imshow('image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def visualize_class(path,clas='road'):
    img=cv2.imread(path)
    clas=np.where(img==classes[clas],255,0).astype(np.uint8)
    cv2.imshow("test1", clas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    path='/nvidia/output/847051_gtFine_polygons_labelTrainIds.png'
    visualize_class(path)
