import pandas as pd
import cv2
path_base="./images/"
df=pd.read_csv('metadata.csv')
df=df[df['view']=='PA'][df['modality']=='X-ray']
list_im_covid=df[df['finding']=='COVID-19']['filename'].tolist()
list_im_n_covid=df[~(df['finding']=='COVID-19')]['filename'].tolist()
for file in list_im_covid:
    cv2.imwrite("./covid/"+file,cv2.imread(path_base+file))

for file in list_im_n_covid:
    cv2.imwrite("./no_covid/"+file,cv2.imread(path_base+file))
