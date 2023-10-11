from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Embedding, Flatten, Concatenate, LeakyReLU
import tensorflow.keras.backend as K

def build_conditional_discriminator():
    data = Input(shape=(30,))
    label = Input(shape=(1,), dtype='int32')
    label_embedding = Embedding(2, 30)(label)
    label_embedding = Flatten()(label_embedding)
    merged_input = Concatenate(axis=-1)([data, label_embedding])
    x = Dense(128)(merged_input)
    x = LeakyReLU(alpha=0.2)(x)
    out = Dense(1)(x)
    return Model([data, label], out)