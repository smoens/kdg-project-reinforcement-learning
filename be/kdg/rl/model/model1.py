from keras import layers
from keras import models
from keras import optimizers

from be.kdg.rl.utils import config

def create_model(selected_model, state_size, action_size):
    ############# MODEL #############
    model = models.Sequential()
    model.add(layers.Dense(256, activation='relu', input_dim=state_size))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(action_size, activation='linear'))

    model.summary()

    model.compile(loss='mse',
                  optimizer=optimizers.RMSprop(lr=0.00025, rho=0.95, epsilon=0.01),
                  #optimizer=optimizers.Adam(lr=1e-4),
                  metrics=config.params.get("models").get(selected_model).get("metrics"))

    return model