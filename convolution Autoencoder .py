from keras.layers import Conv2D, Conv2DTranspose, Dense, Flatten, Reshape
from keras.models import Sequential, Model
import numpy as np
from PIL import Image
from os import listdir
import os
from os.path import isfile, join
import cv2
from tqdm import tqdm
import pathlib
import itertools
import matplotlib.pyplot as plt

#  Getting data
search_images=[]
test_images= []
def load_searchdata():
        search_data=[]
        i=0
        for img in tqdm(os.listdir('Images/search/images')):
                path=os.path.join ('Images/search/images/',img)
                image=cv2.imread(path,cv2.IMREAD_GRAYSCALE)                # cv2.IMREAD_COLOR
                search_images.append(img)
                pathlib.Path('C:/Users/ali/Desktop/matching_images/'+img).mkdir(parents=True, exist_ok=True) 
                i+=1
                try:
                        im=Image.open(path)
                        im.verify()
                except(IOError, SyntaxError) as e:
                                print('Bad file:'+ image)
                image=cv2.resize(image, (28,28)).astype('float32')/255
                search_data.append(image)
        return search_data

def load_testdata():
        test_data=[]
        for img in tqdm(os.listdir('Images/test/images')):
                path=os.path.join ('Images/test/images/',img)
                image=cv2.imread(path,cv2.IMREAD_GRAYSCALE)
                test_images.append(img)
                try:
                        im=Image.open(path)
                        im.verify()
                except(IOError, SyntaxError) as e:
                                print('Bad file:'+ image)
                image=cv2.resize(image, (28,28)).astype('float32')/255
                test_data.append(image)
        return test_data

def load_traindata():
        train_data=[]
        for img in tqdm(os.listdir('Images/train/imagesss')):
                path=os.path.join ('Images/train/imagesss/',img)
                image=cv2.imread(path,cv2.IMREAD_GRAYSCALE)
                try:
                        im=Image.open(path)
                        im.verify()
                except(IOError, SyntaxError) as e:
                                print('Bad file:'+ image)
                image=cv2.resize(image, (28,28)).astype('float32')/255
                train_data.append(image)
        return train_data

search_data=load_searchdata()
test_data=load_testdata()
train_data=load_traindata()


test_data= np.reshape(test_data,(-1,28,28,1)).astype('float32')
search_data=np.reshape(search_data,(-1,28,28,1)).astype('float32')

train_data=np.reshape(train_data,(-1,28,28,1)).astype('float32')

#train=train_data[:100000]
#train= np.reshape(train,(-1,28,28,3)).astype('float32')/255





 # Constructing Convolutional Autoencoder 



def CAE(input_shape=(28, 28, 1), filters=[32, 64, 128, 10]):
    model = Sequential()
    if input_shape[0] % 8 == 0:
        pad3 = 'same'
    else:
        pad3 = 'valid'
    model.add(Conv2D(filters[0], 5, strides=2, padding='same', activation='relu', name='conv1', input_shape=input_shape))

    model.add(Conv2D(filters[1], 5, strides=2, padding='same', activation='relu', name='conv2'))

    model.add(Conv2D(filters[2], 3, strides=2, padding=pad3, activation='relu', name='conv3'))

    model.add(Flatten())
    model.add(Dense(units=filters[3], name='embedding'))
    model.add(Dense(units=filters[2]*int(input_shape[0]/8)*int(input_shape[0]/8), activation='relu'))

    model.add(Reshape((int(input_shape[0]/8), int(input_shape[0]/8), filters[2])))
    model.add(Conv2DTranspose(filters[1], 3, strides=2, padding=pad3, activation='relu', name='deconv3'))

    model.add(Conv2DTranspose(filters[0], 5, strides=2, padding='same', activation='relu', name='deconv2'))

    model.add(Conv2DTranspose(input_shape[2], 5, strides=2, padding='same', name='deconv1'))
    model.summary()
    return model

if __name__ == "__main__":
    from time import time

    # setting the hyper parameters
    import argparse
    parser = argparse.ArgumentParser(description='train')
    parser.add_argument('--dataset')
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--epochs', default=50, type=int)
    args = parser.parse_args()
    print(args)


    x=train_data
  

    # define the model
    model = CAE(input_shape=x.shape[1:], filters=[32, 64, 128, 10])
    model.summary()

    # compile the model and callbacks
    optimizer = 'adam'
    model.compile(optimizer=optimizer, loss='mse')


    # begin training
    t0 = time()
    model.fit(x, x, batch_size=args.batch_size, epochs=args.epochs)


    # extract features
    feature_model = Model(inputs=model.input, outputs=model.get_layer(name='embedding').output)




    # Predicting train_data features
    
    traindata_features = feature_model.predict(train_data)     # Pass here test data
    print('feature shape=', traindata_features.shape)

     # CLUSTERING   
    from sklearn.cluster import KMeans
    kmeans=KMeans(n_clusters=10,init='k-means++',max_iter=300,n_init=10,random_state=0)
    y_kmeans=kmeans.fit_predict(traindata_features)
    
    
    
    #Using the elbow method to find the exact number of clusters
    from sklearn.cluster import KMeans
    wcss=[] #list
    for i in range(1,11): #loop for 10 clusters
          kmeans=KMeans(n_clusters=i,init= 'k-means++',max_iter=300,n_init= 10,random_state=0)
          kmeans.fit(traindata_features)
          wcss.append(kmeans.inertia_) # clusters sum of squares computing and adding in list

    plt.plot(range(1,11),wcss)  # X-axis and y-axis values
    plt.title('The ELbow Method')
    plt.xlabel('Number of Clusters')
    plt.ylabel('WCSS')
    plt.show()
 
#========================================================

     # Getting Seach img features
    
    search_imgfeatures=[len(search_data)]
    for i in range(len(search_data)):
            search= search_data[i]
            search=search.reshape(-1,28,28,3).astype('float32')   
            imgfeatures=feature_model.predict(search)    
            search_imgfeatures.append(imgfeatures)
     
        
        
     # Get index of 100 images near Centroid 
    from scipy import spatial
    tree= spatial.KDTree(testdata_features)
         
    def get_topimages(fno):
            indexes=[]
            d,ind=tree.query(fno,k=100)
            indexes.append(ind)
            return indexes
        
    # Save images     
        
     def save_data():        
        links=[] 
        for img in tqdm(os.listdir('Images/test/images')):
                path=os.path.join ('Images/test/images/',img)
                links.append(path)
        for k in range(1,528):
                feature_no= search_imgfeatures[k]
                top_im= get_topimages(feature_no)
                name= search_images[k-1] 
                j=0
                for item in itertools.chain.from_iterable(top_im): 
                        while j<=99:
                                i=item[j]
                                image=cv2.imread(links[i],cv2.IMREAD_COLOR)
                                cv2.imwrite('C:/Users/ali/Desktop/matching_images/'+name+'/'+str(j)+'_'+test_images[i]+'.jpg',image)
                                j+=1
                                print(j)
                        print('Done')

    save_data()    
        
        
        
        
        
         
    
      
                 
