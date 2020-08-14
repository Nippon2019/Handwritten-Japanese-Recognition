import skimage.transform
import numpy as np
from sklearn.model_selection import train_test_split

hira = np.load("hiragana.npz")['arr_0'].reshape([-1, 127, 128]).astype(np.float32)

hira = hira/np.max(hira)

# 71 characters, 160 writers, transform image to 48*48
train_images = np.zeros([71 * 160, 48, 48], dtype=np.float32)

for i in range(71 * 160):
    train_images[i] = skimage.transform.resize(hira[i], (48, 48))

arr = np.arange(71)
train_labels = np.repeat(arr, 160) # create labels

# split to train and test
train_images, test_images, train_labels, test_labels = train_test_split(train_images, train_labels, test_size=0.2)

np.savez_compressed("hiragana_train_images.npz", train_images)
np.savez_compressed("hiragana_train_labels.npz", train_labels)
np.savez_compressed("hiragana_test_images.npz", test_images)
np.savez_compressed("hiragana_test_labels.npz", test_labels)
