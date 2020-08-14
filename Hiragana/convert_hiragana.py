from hiraganajapanese import label

import coremltools
import keras
from keras.utils import CustomObjectScope
from keras.initializers import glorot_uniform
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

scale = 1/255.0

with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
	hiragana_model = coremltools.converters.keras.convert("./hiragana.h5",input_names=["image"],image_input_names = 'image',class_labels=label,image_scale=scale)
hiragana_model.author = "Aiyu Kamate"
hiragana_model.short_description = "Handwritten Hiragana Recognition"
hiragana_model.input_description['image'] = 'Detects handwritten Hiragana'
hiragana_model.output_description['output'] = 'Prediction of Hiragana'
hiragana_model.save("hiragana.mlmodel")
