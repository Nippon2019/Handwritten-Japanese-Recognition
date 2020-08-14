from kanjijapanese import label

import coremltools
import keras
from keras.utils import CustomObjectScope
from keras.initializers import glorot_uniform
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

scale = 1/255.0

with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
	kanji_model = coremltools.converters.keras.convert("./kanji.h5",input_names=["image"],image_input_names = 'image',class_labels=label,image_scale=scale)
kanji_model.author = "Aiyu Kamate"
kanji_model.short_description = "Handwritten Kanji Recognition"
kanji_model.input_description['image'] = 'Detects handwritten Kanji'
kanji_model.output_description['output'] = 'Prediction of Kanji'
kanji_model.save("kanji.mlmodel")
