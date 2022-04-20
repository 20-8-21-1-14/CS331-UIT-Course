import numpy as np
from keras.models import Model
from matplotlib import pyplot as plt
from tensorflow.keras.utils import load_img
from keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg16 import VGG16
  
model = VGG16(weights="imagenet", include_top=True)

print(model.summary())

#### Now plot filter outputs    

#Define a new truncated model to only include the conv layers of interest
"""
    Conv1, Pooling 1, Normalize 1 (nếu có)
    Conv cuối cùng.
"""

def show_and_save_feature(feature_output, name_layer):
    columns = 8
    rows = 8
    for ftr in feature_output:
        #pos = 1
        fig=plt.figure(figsize=(12, 12))
        for i in range(1, columns*rows +1):
            fig =plt.subplot(rows, columns, i)
            fig.set_xticks([])  #Turn off axis
            fig.set_yticks([])
            plt.imshow(ftr[:, :, i-1], cmap='gray')
            #pos += 1
        plt.savefig(name_layer+'.png', bbox_inches='tight')
        plt.show()


if __name__ == "__main__":
    conv_layer_index = [1, 3, 17]  #TO define a shorter model
    outputs = [model.layers[i].output for i in conv_layer_index]
    model_short = Model(inputs=model.inputs, outputs=outputs)
    print(model_short.summary())

    #Input shape to the model is 224 x 224. SO resize input image to this shape.
    from keras.preprocessing.image import load_img, img_to_array
    img = load_img('Me_0.jpg', target_size=(224, 224)) #VGG user 224 as input

    # convert the image to an array
    img = img_to_array(img)
    # expand dimensions to match the shape of model input
    img = np.expand_dims(img, axis=0)

    # Generate feature output by predicting on the input image
    # feature_output_1 get 1st conv layer
    # feature_output_2 get pooling layer
    # feature_output_3 get last conv layer
    feature_output_1 = model_short.predict(img)[0]
    feature_output_2 = model_short.predict(img)[1]
    feature_output_3 = model_short.predict(img)[2]
    show_and_save_feature(feature_output_1, name_layer='First_conv')
    show_and_save_feature(feature_output_2, name_layer='Max_pooling')
    show_and_save_feature(feature_output_3, name_layer='Last_conv')