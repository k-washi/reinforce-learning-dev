import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as kl
import tensorflow_probability as tfp

class PolicyWithValue(tf.keras.Model):

    def __init__(self, action_space):
        """ PolicyとValueが一部ネットワークを共有するA3Cアーキテクチャ
        """
        super(PolicyWithValue, self).__init__()
        self.dense1 = kl.Dense(64, activation="relu")
        self.dense2_1 = kl.Dense(64, activation="relu")
        self.dense2_2 = kl.Dense(64, activation="relu")
        self.values = kl.Dense(1)
        self.logits = kl.Dense(action_space)

    @tf.function
    def call(self, x):
        x = self.dense1(x)
        
        x1 = self.dense2_1(x)
        logits = self.logits(x1)
        action_probs = tf.nn.softmax(logits)
        
        x2 = self.dense2_2(x)
        values = self.values(x2)

        return values, action_probs

    def sample_action(self, state):
        state = np.atleast_2d(state) # [[2]]
        _, action_probs = self(state) # tf.Tensor([[0.19595419 0.20959526 0.14920488 0.21499905 0.2302467 ]], shape=(1, 5), dtype=float32)
        cdist = tfp.distributions.Categorical(probs=action_probs) # tfp.distributions.Categorical("Categorical", batch_shape=[1], event_shape=[], dtype=int32)
        action = cdist.sample() # tf.Tensor([2], shape=(1,), dtype=int32)
        return action.numpy()[0]

if __name__ == "__main__":
    model = PolicyWithValue(5)
    state = 2
    res = model.sample_action(state)
    print(res)