#All imports
import matplotlib.pyplot as plt,cv2,os,numpy as np,pandas as pd,seaborn as sn
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16,InceptionV3
from tensorflow.keras.layers import AveragePooling2D,Dropout,Input,Flatten,Dense,MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

#Hyper parametters
lr=1e-3;epochs=100;batch_size=8;
y=[]
x=[]

def get_images(path):
    list_images=[]
    for (root, _, files) in os.walk(path):
        for file in files:
            if(file.split('.')[-1] in ["bmp", "jpeg", "png", "jpg"]):
                list_images.append(os.path.join(root, file))
    return list_images

def get_arrays(path,in_size=(224,224),true_path="covid"):
    x_=[];y_=[]
    for f_name in get_images(path):
        data=np.array(cv2.resize(cv2.imread(f_name)[:,:,::-1],in_size,interpolation=cv2.INTER_CUBIC))
        data=data/255.0;#data-=0.5;data*=2;
        x_.append(data)
        y_.append(1 if f_name.split('/')[-2]==true_path else 0)
    return  np.array(x_),to_categorical(y_)


def get_model(dp=0.5,labels=2):
    Model_base=VGG16(include_top=False,weights="imagenet",input_tensor=Input(shape=(224, 224, 3)))
    x=Model_base.output
    x = MaxPooling2D(pool_size=(4,4))(x)
    x=Flatten()(x)
    x=Dense(64,activation='relu')(x)
    x = Dropout(dp)(x)
    x = Dense(64, activation='relu')(x)
    x=Dense(labels,activation='softmax')(x)
    model=Model(inputs=Model_base.input,outputs=x)
    for layer in Model_base.layers:layer.trainable=False
    return model

def get_data_augumentation(X,Y,batch_siz):
    Generator=ImageDataGenerator(rotation_range=15,fill_mode='nearest')
    return Generator.flow(X,Y,batch_size=batch_siz)


x,y=get_arrays('./dataset')
X_train,X_test,Y_train,Y_test=train_test_split(x, y,test_size=0.20, random_state=1)
covid_detector=get_model()
covid_detector.compile(loss='binary_crossentropy',optimizer=Adam(lr=lr,decay=lr/epochs),metrics=['accuracy'])
covid_detector.summary()
history=covid_detector.fit_generator(get_data_augumentation(X_train,Y_train,batch_size),steps_per_epoch=len(X_train)//batch_size,epochs=epochs,validation_data=(X_test,Y_test),validation_steps=len(X_test)//batch_size)
ys = np.argmax(covid_detector.predict(X_test, batch_size=batch_size),axis=1)
Conf_matrix = confusion_matrix(Y_test.argmax(axis=1), ys)
df_cm = pd.DataFrame(Conf_matrix, range(2), range(2))
sn.set(font_scale=1.4)
sn.heatmap(df_cm, annot=True, annot_kws={"size": 16})
plt.show()
covid_detector.save('covid.h5')












