from keras import layers
from keras import models
from keras import optimizers

from be.kdg.rl.config import config

def createModel(selectedModel):
    ############# MODEL #############
    model = models.Sequential()
    model.add(layers.Dense(256, activation='relu', input_shape=[]))
        # TODO add correct input spaces...
    model.add(layers.Dense(64, activation='relu', input_shape=[]))
        # TODO add layer size of action_size
    #model.add(layers.Dense(action_size, activation='softmax'))

    model.summary()

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.RMSprop(lr=1e-4),
                  metrics=config.params.get(selectedModel).get("metrics"))

    return model
