# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 01:59:23 2018

@author: ali
"""


from __future__ import division, print_function, unicode_literals
import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf 

# reset the default graph
tf.reset_default_graph() 

#set the random seeds for always produceing the same output:
np.random.seed(42)
tf.set_random_seed(42)

#===========================================================
 # Import data here !




y_label= y_kmeans.astype(np.int64)
X_data=train_data


## Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_data, y_label, test_size = 0.2, random_state = 0)
X_train, X_validation, y_train, y_validation     = train_test_split(X_train, y_train, test_size=0.2, random_state=1)






#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets("/tmp/data/")
#
#n_samples = 5
#plt.figure(figsize=(n_samples * 2, 3))
#for index in range(n_samples):
#    plt.subplot(1, n_samples, index + 1)
#    sample_image = mnist.train.images[index].reshape(28, 28)
#    plt.imshow(sample_image, cmap="binary")
#    plt.axis("off")
#
#plt.show()
#
##And these are the corresponding labels:
#mnist.train.labels[:n_samples]


#creating a placeholder for the input images (28×28 pixels).
#batch size is None so we can input any no of images, 
#image dimension, 3 for colour images
X = tf.placeholder(shape=[None, 28, 28, 1], dtype=tf.float32, name="X") # change image dimension and check results, 3 for color images, 1 for gray
y = tf.placeholder(shape=[None], dtype=tf.int64, name="y")
#Primary Capsules
#The first layer will be composed of 32 maps of 6×6 capsules each, where each capsule will output an 8D activation vector

caps1_n_maps = 32
caps1_n_caps = caps1_n_maps * 6 * 6  # 1152 primary/1st layer capsules
caps1_n_dims = 8

#To compute capsules outputs, we first apply two regular convolutional layers:
       # this is how we end up with 6×6 feature maps
conv1_params = {
    "filters": 256,
    "kernel_size": 9,
    "strides": 1,
    "padding": "valid",
    "activation": tf.nn.relu,
}
 # 2nd layer is configured to output 256(8*32) feature maps
conv2_params = {
    "filters": caps1_n_maps * caps1_n_dims, # 256 convolutional filters
    "kernel_size": 9,
    "strides": 2,
    "padding": "valid",
    "activation": tf.nn.relu
}

conv1 = tf.layers.conv2d(X, name="conv1", **conv1_params) #input of 1st layer is x (placeholder containing input images feeded at runtime)
#takes output of 1st layer and parameters specified in Paper
conv2 = tf.layers.conv2d(conv1, name="conv2", **conv2_params)

#reshape the output to get a bunch of 8D vectors representing the outputs of the primary capsules
caps1_raw = tf.reshape(conv2, [-1, caps1_n_caps, caps1_n_dims],
                       name="caps1_raw")

#apply this function to get the output $\mathbf{u}_i$ of each primary capsules according to Paper
# squash to make sure ,Length of feature vectors is b/w 0 and 1 
def squash(s, axis=-1, epsilon=1e-7, name=None):
    with tf.name_scope(name, default_name="squash"):
        squared_norm = tf.reduce_sum(tf.square(s), axis=axis,
                                     keep_dims=True)
        safe_norm = tf.sqrt(squared_norm + epsilon)
        squash_factor = squared_norm / (1. + squared_norm)
        unit_vector = s / safe_norm
        return squash_factor * unit_vector
#output of the first capsule layer
caps1_output = squash(caps1_raw, name="caps1_output")


# Compute the Predicted Output Vectors
#2nd capsule layer contains 10 capsules (one for each digit) of 16 dimensions each
caps2_n_caps = 10
caps2_n_dims = 16

#For each capsule $i$ in the first layer, we want to predict the output of every capsule $j$ in the second layer
init_sigma = 0.01

W_init = tf.random_normal(
    shape=(1, caps1_n_caps, caps2_n_caps, caps2_n_dims, caps1_n_dims),
    stddev=init_sigma, dtype=tf.float32, name="W_init")
W = tf.Variable(W_init, name="W")

batch_size = tf.shape(X)[0]
W_tiled = tf.tile(W, [batch_size, 1, 1, 1, 1], name="W_tiled")

caps1_output_expanded = tf.expand_dims(caps1_output, -1,
                                       name="caps1_output_expanded")
caps1_output_tile = tf.expand_dims(caps1_output_expanded, 2,
                                   name="caps1_output_tile")
caps1_output_tiled = tf.tile(caps1_output_tile, [1, 1, caps2_n_caps, 1, 1],
                             name="caps1_output_tiled")
# check the shape of the first array
W_tiled
# 2nd array shape
caps1_output_tiled

#to get all the predicted output vectors, we just need to multiply these two arrays using tf.matmul()
caps2_predicted = tf.matmul(W_tiled, caps1_output_tiled,
                            name="caps2_predicted")
caps2_predicted

# Routing by agreement

raw_weights = tf.zeros([batch_size, caps1_n_caps, caps2_n_caps, 1, 1],
                       dtype=np.float32, name="raw_weights")
#apply the softmax function to compute the routing weights
routing_weights = tf.nn.softmax(raw_weights, dim=2, name="routing_weights")
#compute the weighted sum of all the predicted output vectors for each second-layer capsule
weighted_predictions = tf.multiply(routing_weights, caps2_predicted,
                                   name="weighted_predictions")
weighted_sum = tf.reduce_sum(weighted_predictions, axis=1, keep_dims=True,
                             name="weighted_sum")



caps2_output_round_1 = squash(weighted_sum, axis=-2,
                              name="caps2_output_round_1")
caps2_output_round_1

caps2_predicted
caps2_output_round_1

caps2_output_round_1_tiled = tf.tile(
    caps2_output_round_1, [1, caps1_n_caps, 1, 1, 1],
    name="caps2_output_round_1_tiled")
agreement = tf.matmul(caps2_predicted, caps2_output_round_1_tiled,
                      transpose_a=True, name="agreement")

raw_weights_round_2 = tf.add(raw_weights, agreement,
                             name="raw_weights_round_2")


routing_weights_round_2 = tf.nn.softmax(raw_weights_round_2,
                                        dim=2,
                                        name="routing_weights_round_2")
weighted_predictions_round_2 = tf.multiply(routing_weights_round_2,
                                           caps2_predicted,
                                           name="weighted_predictions_round_2")
weighted_sum_round_2 = tf.reduce_sum(weighted_predictions_round_2,
                                     axis=1, keep_dims=True,
                                     name="weighted_sum_round_2")
caps2_output_round_2 = squash(weighted_sum_round_2,
                              axis=-2,
                              name="caps2_output_round_2")

caps2_output = caps2_output_round_2


#Static or Dynamic Loop?

#def condition(input, counter):
#    return tf.less(counter, 100)
#
#def loop_body(input, counter):
#    output = tf.add(input, tf.square(counter))
#    return output, tf.add(counter, 1)
#
#with tf.name_scope("compute_sum_of_squares"):
#    counter = tf.constant(1)
#    sum_of_squares = tf.constant(0)
#
#    result = tf.while_loop(condition, loop_body, [sum_of_squares, counter])
#    
#
#with tf.Session() as sess:
#    print(sess.run(result))
#    
#    
#sum([i**2 for i in range(1, 100 + 1)])


#Estimated Class Probabilities (Length)
def safe_norm(s, axis=-1, epsilon=1e-7, keep_dims=False, name=None):
    with tf.name_scope(name, default_name="safe_norm"):
        squared_norm = tf.reduce_sum(tf.square(s), axis=axis,
                                     keep_dims=keep_dims)
        return tf.sqrt(squared_norm + epsilon)

y_proba = safe_norm(caps2_output, axis=-2, name="y_proba")
y_proba_argmax = tf.argmax(y_proba, axis=2, name="y_proba")
y_proba_argmax


y_pred = tf.squeeze(y_proba_argmax, axis=[1,2], name="y_pred")
y_pred



#Margin loss (paper)
m_plus = 0.9
m_minus = 0.1
lambda_ = 0.5

T = tf.one_hot(y, depth=caps2_n_caps, name="T")
with tf.Session():
    print(T.eval(feed_dict={y: np.array([0, 1, 2, 3, 9])}))
# 2nd capsule layer output    
caps2_output

caps2_output_norm = safe_norm(caps2_output, axis=-2, keep_dims=True,
                              name="caps2_output_norm")

present_error_raw = tf.square(tf.maximum(0., m_plus - caps2_output_norm),
                              name="present_error_raw")
present_error = tf.reshape(present_error_raw, shape=(-1, 10),
                           name="present_error")
absent_error_raw = tf.square(tf.maximum(0., caps2_output_norm - m_minus),
                             name="absent_error_raw")
absent_error = tf.reshape(absent_error_raw, shape=(-1, 10),
                          name="absent_error")


L = tf.add(T * present_error, lambda_ * (1.0 - T) * absent_error,
           name="L")

margin_loss = tf.reduce_mean(tf.reduce_sum(L, axis=1), name="margin_loss")


#Reconstruction
# add a decoder network on top of the capsule network
mask_with_labels = tf.placeholder_with_default(False, shape=(),
                                               name="mask_with_labels")

reconstruction_targets = tf.cond(mask_with_labels, # condition
                                 lambda: y,        # if True
                                 lambda: y_pred,   # if False
                                 name="reconstruction_targets")

reconstruction_mask = tf.one_hot(reconstruction_targets,
                                 depth=caps2_n_caps,
                                 name="reconstruction_mask")

reconstruction_mask
caps2_output
reconstruction_mask_reshaped = tf.reshape(
    reconstruction_mask, [-1, 1, caps2_n_caps, 1, 1],
    name="reconstruction_mask_reshaped")

caps2_output_masked = tf.multiply(
    caps2_output, reconstruction_mask_reshaped,
    name="caps2_output_masked")
caps2_output_masked

decoder_input = tf.reshape(caps2_output_masked,
                           [-1, caps2_n_caps * caps2_n_dims],
                           name="decoder_input")

decoder_input

#Decoder
# build the decoder. It'has two dense (fully connected) ReLU layers followed by a dense output sigmoid layer
n_hidden1 = 512
n_hidden2 = 1024
n_output = 28 * 28

with tf.name_scope("decoder"):
    hidden1 = tf.layers.dense(decoder_input, n_hidden1,
                              activation=tf.nn.relu,
                              name="hidden1")
    hidden2 = tf.layers.dense(hidden1, n_hidden2,
                              activation=tf.nn.relu,
                              name="hidden2")
    decoder_output = tf.layers.dense(hidden2, n_output,
                                     activation=tf.nn.sigmoid,
                                     name="decoder_output")
  # decoder_prob= tf.layers.dense(decoder_output,10,activation=tf.nn.sigmoid)
   # Use decoder_probe further into capsule I guess
    
#Reconstruction Loss    
X_flat = tf.reshape(X, [-1, n_output], name="X_flat")
squared_difference = tf.square(X_flat - decoder_output,name="squared_difference")
reconstruction_loss = tf.reduce_sum(squared_difference,
                                    name="reconstruction_loss")
#Final Loss
alpha = 0.0005
loss = tf.add(margin_loss, alpha * reconstruction_loss, name="loss")

#Accuracy

correct = tf.equal(y, y_pred, name="correct")
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")
#Training Operations
optimizer = tf.train.AdamOptimizer()
training_op = optimizer.minimize(loss, name="training_op")
#Init and Saver
init = tf.global_variables_initializer()
saver = tf.train.Saver()




 # function to get next_batch 
 
def next_batch(num, data, labels):
#    '''
#    Return a total of `num` random samples and labels. 
#    '''
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[ i] for i in idx]
    labels_shuffle = [labels[ i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)



#Training Network
n_epochs = 5 # 10
batch_size = 50
restore_checkpoint = True
############################### Made Change here ##############
n_iterations_per_epoch = len(X_train)// batch_size
n_iterations_validation = len(X_validation)// batch_size

#mnist.train.num_examples//batch_size
#mnist.validation.num_examples
best_loss_val = np.infty
checkpoint_path = "./my_capsule_network"

with tf.Session() as sess:
    if restore_checkpoint and tf.train.checkpoint_exists(checkpoint_path):
        saver.restore(sess, checkpoint_path)
    else:
        init.run()

    for epoch in range(n_epochs):
        for iteration in range(1, n_iterations_per_epoch + 1):
            X_batch, y_batch =  next_batch(batch_size,X_train,y_train)  #mnist.train.next_batch(batch_size) 
            # Run the training operation and measure the loss:
            _, loss_train = sess.run(
                [training_op, loss],
                feed_dict={X: X_batch.reshape([-1, 28, 28, 1]),
                           y: y_batch,
                           mask_with_labels: True})        
            print("\rIteration: {}/{} ({:.1f}%)  Loss: {:.5f}".format(
                      iteration, n_iterations_per_epoch,
                      iteration * 100 / n_iterations_per_epoch,
                      loss_train),
                  end="")

        # At the end of each epoch,
        # measure the validation loss and accuracy:
        loss_vals = []
        acc_vals = []
        for iteration in range(1, n_iterations_validation + 1):
            X_batch, y_batch =  next_batch(batch_size, X_validation, y_validation) #mnist.validation.next_batch(batch_size)  
            loss_val, acc_val = sess.run(
                    [loss, accuracy],
                    feed_dict={X: X_batch.reshape([-1, 28, 28, 1]),
                               y: y_batch})
            loss_vals.append(loss_val)
            acc_vals.append(acc_val)
            print("\rEvaluating the model: {}/{} ({:.1f}%)".format(
                      iteration, n_iterations_validation,
                      iteration * 100 / n_iterations_validation),
                  end=" " * 10)
        loss_val = np.mean(loss_vals)
        acc_val = np.mean(acc_vals)
        print("\rEpoch: {}  Val accuracy: {:.4f}%  Loss: {:.6f}{}".format(
            epoch + 1, acc_val * 100, loss_val,
            " (improved)" if loss_val < best_loss_val else ""))

        # And save the model if it improved:
        if loss_val < best_loss_val:
            save_path = saver.save(sess, checkpoint_path)
            best_loss_val = loss_val
            
            #-------- changed here --------------
n_iterations_test = len(X_test)// batch_size
# mnist.test.num_examples
with tf.Session() as sess:
    saver.restore(sess, checkpoint_path)

    loss_tests = []
    acc_tests = []
    for iteration in range(1, n_iterations_test + 1):
        X_batch, y_batch =   next_batch(batch_size,X_test,y_test) #mnist.test.next_batch(batch_size)  
        loss_test, acc_test = sess.run(
                [loss, accuracy],
                feed_dict={X: X_batch.reshape([-1, 28, 28, 1]),
                           y: y_batch})
        loss_tests.append(loss_test)
        acc_tests.append(acc_test)
        print("\rEvaluating the model: {}/{} ({:.1f}%)".format(
                  iteration, n_iterations_test,
                  iteration * 100 / n_iterations_test),
              end=" " * 10)
    loss_test = np.mean(loss_tests)
    acc_test = np.mean(acc_tests)
    print("\rFinal test accuracy: {:.4f}%  Loss: {:.6f}".format(
        acc_test * 100, loss_test))
    
    
    
    
    
    
    # Get test and search images features Manually!   
    

search_features=[]
test_features=[]

# get sample_images for search and test data accordingly 
# 200 range is fine to get features. 
sample_images = test_data[0:200].reshape([-1, 28, 28, 1])
with tf.Session() as sess:
    saver.restore(sess, checkpoint_path)
    caps2_output_value, decoder_output_value, y_pred_value = sess.run(
            [caps2_output, decoder_output, y_pred],feed_dict={X: sample_images, y: np.array([], dtype=np.int64)})
        
    
    
    caps2_output_value.shape


    Tfeatures=np.concatenate([Tfeatures,caps2_output_val])
    Tfeatures=Tfeatures[0:9928]    
    #Concat for search and test data accordingly 
    Sfeatures=np.concatenate([Sfeatures,caps2_output_val])
    
    
    # Sfeatures and Tfeatues must be assigned first 50 values using above code then 
    # get features with down code 

    i=50
    j=100
    with tf.Session() as sess:
            saver.restore(sess, checkpoint_path)
            for k in range(0,197):
                    sample_images = test_data[i:j].reshape([-1, 28, 28, 1])
                    i+=50
                    j+=50
                    caps2_output_val, decoder_output_value, y_pred_value = sess.run(
                                    [caps2_output, decoder_output, y_pred],feed_dict={X: sample_images, y: np.array([], dtype=np.int64)})
                    Tfeatures=np.concatenate([Tfeatures,caps2_output_val]) 
#                        
#caps2_output_val.shape   




    #Reshaping 
    
    Sfeatures= Sfeatures.reshape(527,-1)
    Tfeatures= Tfeatures.reshape(9928,-1)

    
    
     # Get index of 100 images near Centroid 
    from scipy import spatial
    tree= spatial.KDTree(Tfeatures)
         
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
        for k in range(0,527):
                feature_no= Sfeatures[k]
                top_im= get_topimages(feature_no)
                name= search_images[k] 
                j=0
                for item in itertools.chain.from_iterable(top_im): 
                        image=cv2.imread(links[item],cv2.IMREAD_COLOR)
                        cv2.imwrite('C:/Users/ali/Desktop/matching_images/'+name+'/'+str(j)+'_'+test_images[i]+'.jpg',image)
                        j+=1
                        print(j)
                print('Done')

    save_data()   
    
    
    
    