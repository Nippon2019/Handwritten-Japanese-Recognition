from tensorflow import keras
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K

# downloaded from KMNIST
train_images = np.load("k49-train-imgs.npz")['arr_0']
train_labels = np.load("k49-train-labels.npz")['arr_0']
test_images = np.load("k49-test-imgs.npz")['arr_0']
test_labels = np.load("k49-test-labels.npz")['arr_0']

if K.image_data_format() == "channels_first":
  train_images = train_images.reshape(train_images.shape[0], 1,28,28)
  test_images = test_images.reshape(test_images.shape[0], 1,28,28)
  shape = (1,28,28)
else:
  train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)
  test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)
  shape = (28,28,1)

train_images = train_images/255.0

test_images = test_images / 255.0

datagen = ImageDataGenerator(rotation_range=15,zoom_range=0.2)
datagen.fit(train_images)
model = keras.Sequential([
	keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=shape),
	keras.layers.MaxPooling2D(2,2),
	keras.layers.Conv2D(64, (3,3), activation='relu'),
	keras.layers.MaxPooling2D(2,2),
  keras.layers.Flatten(),
  keras.layers.Dropout(0.5),
  keras.layers.Dense(512, activation='relu'),
  keras.layers.Dense(49, activation="softmax")
])

#model.summary()

model.compile(optimizer='adam', loss="sparse_categorical_crossentropy", metrics=['accuracy'])

model.fit_generator(datagen.flow(train_images,train_labels,shuffle=True),epochs=50,validation_data=(test_images,test_labels),callbacks = [keras.callbacks.EarlyStopping(patience=8,verbose=1,restore_best_weights=True),keras.callbacks.ReduceLROnPlateau(factor=0.5,patience=3,verbose=1)])


#test_loss, test_acc = model.evaluate(test_images2, test_labels)
#print("Accuracy: ", test_acc)

model.save("kuzushiji.h5")
