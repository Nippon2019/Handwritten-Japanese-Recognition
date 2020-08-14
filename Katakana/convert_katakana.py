import coremltools
from keras.utils import CustomObjectScope
from keras.initializers import glorot_uniform

import tensorflow.compat.v1 as tf # compat tensorflow 1
tf.disable_v2_behavior()

scale = 1/255.0 # scale from 0 to 1

from katakanajapanese import label # import label

with CustomObjectScope({'GlorotUniform': glorot_uniform()}): # let keras understand that GlorotUniform and glorot_uniform() are the same
  katakana_model = coremltools.converters.keras.convert("./katakana-model.h5",input_names=["image"],image_input_names = 'image',class_labels=label,image_scale=scale)

katakana_model.author = "Aiyu Kamate"
katakana_model.short_description = "Handwritten Katakana Recognition"
katakana_model.input_description['image'] = 'Detects handwritten Katakana'
katakana_model.output_description['output'] = 'Prediction of Katakana'

katakana_model.save("katakana.mlmodel")
