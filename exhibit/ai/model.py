import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam

from config import Config
from utils import write
import numpy as np


class PGAgent:
    """
    Policy gradient agent
    Partly adapted from https://github.com/keon/policy-gradient/blob/master/pg.py
    """

    def __init__(self, state_size, action_size, name="PGAgent", learning_rate=0.001, structure=(200,), verbose=True):
        """
        Set basic variables and construct the model
        :param state_size: Pixels in flattened input state
        :param action_size: Number of possible action types to output
        :param name: Agent name, used in some graphing/visualizing
        :param learning_rate: Model learning rate
        :param structure: Tuple of integers. a dense hidden layer with n layers is crated for each tuple element n
        """
        self.verbose = verbose
        self.name = name
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.99
        self.learning_rate = learning_rate
        self.states = []
        self.gradients = []
        self.rewards = []
        self.probs = []
        self.structure = structure
        self.train_model, self.infer_model = self._build_model()
        if self.verbose: self.infer_model.summary()
        self.last_state = None
        self.last_hidden_activation = None
        self.last_output = None

    def _build_model(self):
        """
        Helper to construct model with Keras based on configuration
        """
        state_input = Input((self.state_size,))
        hidden_layer_output = Dense(self.structure[0], activation='relu', input_shape=(self.state_size,))(state_input)
        x = hidden_layer_output
        if len(self.structure) > 1:
            for layer in self.structure[1:]:
                x = Dense(layer, activation='relu')(x)
        action_output = Dense(self.action_size, activation='softmax')(x)

        # Model with hidden state output for inference visualization
        infer_model = Model(inputs=state_input, outputs=(action_output, hidden_layer_output))
        opt = Adam(lr=self.learning_rate)
        infer_model.compile(loss='categorical_crossentropy', optimizer=opt)

        # Model without state output for training
        train_model = Model(inputs=state_input, outputs=action_output)
        opt = Adam(lr=self.learning_rate)
        train_model.compile(loss='categorical_crossentropy', optimizer=opt)

        return train_model, infer_model

    def act(self, state, greedy=False):
        """
        Infer action from state
        :param state: ndarray representing game state
        :param greedy: sample only the highest-confidence output (do not use during training)
        :return: (action id, confidence vector)
        """
        state = state.reshape([1, state.shape[0]])
        prob, activation = self.infer_model(state, training=False)
        self.last_hidden_activation = activation.numpy().squeeze()
        self.last_output = prob.numpy().flatten()
        if greedy:
            action = np.argmax(self.last_output)
        else:
            action = np.random.choice(self.action_size, 1, p=self.last_output)[0]
        state_ravel = state.reshape(Config.instance().CUSTOM_STATE_SHAPE)
        self.last_state = state_ravel.flatten()

        return action, None, self.last_output

    def get_structure_packet(self):
        """
        Returns the state of the model suitable for realtime visualization
        :return: Model weights (list of 2d lists), biases (list of 1d lists),
        """
        layers = []
        i = 0
        for w in self.infer_model.weights:
            l = None
            if i == 0: # Rotate first weight matrix as temporary solution for rotated
                l = w.numpy().reshape(*Config.instance().CUSTOM_STATE_SHAPE, -1)
                l = l.reshape(Config.instance().CUSTOM_STATE_SIZE, 200).tolist()
            else:
                l = w.numpy().tolist()

            layers.append(l)
            i += 1
        return layers

    def get_activation_packet(self):
        """
        Returns the state of the model suitable for realtime visualization
        :return: Model weights (list of 2d lists), biases (list of 1d lists),
        """
        import time
        t1 = time.perf_counter()
        # First, get input activations (the last preprocessed state)
        input_activation = self.last_state.tolist()

        # Then, get hidden layer activations (using the truncated model)
        hidden_activations = self.last_hidden_activation.tolist()

        # Finally, store output activations (model prediction)
        output_activations = self.last_output.tolist()

        return [input_activation, hidden_activations, output_activations]


    def discount_rewards(self, rewards):
        """
        "Smears" the reward values back through time so frames leading up to a reward are associated to that reward
        :param rewards: vector representing reward at each frame
        :return: discounted reward vector
        """
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(0, rewards.size)):
            if rewards[t] != 0:
                running_add = 0
            running_add = running_add * self.gamma + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards.astype(np.float32)

    def train(self, states, actions, probs, rewards):
        """
        Train the model on a batch of game data. Imlements the "REINFORCE" algorithm.
        :param states: states from each frame
        :param actions: inferred actions from each frame
        :param probs: confidence probabilities from each frame
        :param rewards: rewards from each frame
        :return:
        """
        gradients = []
        for i in range(len(actions)):
            action = actions[i]
            prob = probs[i]
            y = np.zeros([self.action_size])
            y[action] = 1
            gradients.append(np.array(y).astype('float32') - prob)

        gradients = np.vstack(gradients)
        rewards = np.vstack(rewards)
        rewards = self.discount_rewards(rewards)
        gradients *= rewards
        X = np.squeeze(np.vstack([states]))
        Y = probs + self.learning_rate * np.squeeze(np.vstack([gradients]))

        # It shouldn't be necessary to update the inference model explicitly,
        # since it shares weights with the train model
        result = self.train_model.train_on_batch(X, Y)
        write(str(result), f'analytics/{self.name}.csv')

    def load(self, name):
        """
        Load weights from an h5 file
        :param name: path to load weights
        """
        if self.verbose: print(f"Loading {name}")
        self.train_model.load_weights(name)
        self.infer_model.load_weights(name)

    def save(self, name):
        """
        Export weights to an h5 file
        :param name: path to save weights
        """
        print(f"Saving {name}")
        self.train_model.save_weights(name)

