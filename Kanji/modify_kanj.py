import skimage.transform
import numpy as np
from sklearn.model_selection import train_test_split

kanji = 879
rows = 48
cols = 48

kan = np.load("kanji.npz")['arr_0'].reshape([-1, 127, 128]).astype(np.float32)

kan = kan/np.max(kan)

train_images = np.zeros([kanji * 160, rows, cols], dtype=np.float32)

arr = np.arange(kanji)
train_labels = np.repeat(arr, 160)

# 4 characters were actually hiragana, so delete these 4 extras
for i in range( (kanji+4) * 160):
	if int(i/160) != 88 and int(i/160) != 219 and int(i/160) != 349 and int(i/160) != 457:
		if int(i/160) < 88:
			train_images[i] = skimage.transform.resize(kan[i], (rows, cols))
		if int(i/160) > 88 and int(i/160) < 219:
			train_images[i-160] = skimage.transform.resize(kan[i], (rows, cols))
		if int(i/160) > 219 and int(i/160) < 349:
			train_images[i-320] = skimage.transform.resize(kan[i], (rows, cols))
		if int(i/160) > 349 and int(i/160) < 457:
		if int(i/160) > 457:
			train_images[i-640] = skimage.transform.resize(kan[i], (rows, cols))
      
train_images, test_images, train_labels, test_labels = train_test_split(train_images, train_labels, test_size=0.2)

np.savez_compressed("kanji_train_images.npz", train_images)
np.savez_compressed("kanji_train_labels.npz", train_labels)
np.savez_compressed("kanji_test_images.npz", test_images)
np.savez_compressed("kanji_test_labels.npz", test_labels)
