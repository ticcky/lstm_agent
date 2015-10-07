"""
Reinforcement learning algorithms.
"""

import numpy as np


class EnvironmentFeedback(object):
    """
    Holds information that the environment sends to the agents.

    reward - How much reward has the agent collected by performing the last action?
    state - What's the current state? (can be whatever - an object, tuple, ...)
    state_id - Integer representation of the state.
    """
    def __init__(self, reward, state, state_id, final):
        self.reward = reward
        self.state = state
        self.state_id = state_id
        self.final = final

    def __repr__(self):
        return "r(%.2f) %d %s f:%s" % (self.reward, self.state_id, self.state,
                                       self.final)

    def __eq__(self, other):
        return (self.reward == other.reward
                and self.state == other.state
                and self.state_id == other.state_id
                and self.final == other.final)



class Agent(object):
    """
    Interface of an agent.
    """
    def save(self, filename):
        """Save the agent's state to the given file."""
        raise NotImplementedError()

    def load(self, filename):
        """Load the agent's state from the given file."""
        raise NotImplementedError()

    def run(self):
        """Run the agent, yield actions and read states and rewards.

        This should be a generator function that calls first yield to obtain the
        current state, and return the actions by the subsequent yields."""
        raise NotImplementedError()

    def get_name(self):
        """Return the name of the agent."""
        raise NotImplementedError()


class HumanAgent(Agent):
    """
    Agent that uses raw_input from the console to play.

    The user enters the action number.
    """
    input_fn = raw_input

    def __init__(self, input_fn=None):
        if input_fn:
            self.input_fn = input_fn

    def get_name(self):
        return 'human'

    def run(self):
        yield  # Initialization.
        while True:
            a = None
            while True:
                try:
                    a = int(self.input_fn('Enter the move number:'))
                except:
                    pass
                else:
                    break
            yield a


class QPolicyAgent(object):
    """
    Base class for agents that use the Q-function to play.
    """
    def __init__(self, alpha, gamma, n_actions, n_states, epsilon):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        self.n_actions = n_actions
        self.n_states = n_states

        self.q = np.zeros((self.n_states, self.n_actions))

    def save(self, f_out):
        np.save(f_out, self.q)

    def load(self, f_in):
        self.q = np.load(f_in)

    def get_name(self):
        return "[alpha=%.2f,gamma=%.2f,epsilon=%.2f]" % (self.alpha, self.gamma, self.epsilon)

    def _get_max_action(self, state_id):
        """Get the best action from the given state."""
        qvals = self.q[state_id]
        candidate_actions = np.argwhere(qvals == np.amax(qvals))[:,0]
        return np.random.choice(candidate_actions)

    def _get_action(self, state_id):
        """Get epsilon-greedy action from the given state."""
        if np.random.random() >= self.epsilon:
            return self._get_max_action(state_id)
        else:
            return np.random.random_integers(0, self.n_actions - 1)


class SARSAAgent(QPolicyAgent):
    """
    Agent that implements SARSA control algorithm to play.
    """
    def run(self):
        env_feedback = yield
        s = env_feedback.state_id
        a = self._get_action(s)
        while True:
            env_feedback = yield a
            r, sp = env_feedback.reward, env_feedback.state_id

            ap = self._get_action(sp)

            curr_q = self.q[s, a]
            curr_qp = self.q[sp, ap]
            new_q = curr_q + self.alpha * (r + self.gamma * curr_qp - curr_q)
            self.q[s, a] = new_q

            s = sp
            a = ap

    def get_name(self):
        return "sarsa%s" % super(SARSAAgent, self).get_name()


class QLearningAgent(QPolicyAgent):
    """
    Agent that implements Q-learning algorithm to play.
    """
    def run(self):
        env_feedback = yield
        s = env_feedback.state_id
        while True:
            a = self._get_action(s)
            env_feedback = yield a
            r, sp = env_feedback.reward, env_feedback.state_id

            curr_q = self.q[s, a]

            max_a = self._get_max_action(sp)
            q_max = self.q[sp, max_a]

            new_q = curr_q + self.alpha * (r + self.gamma * q_max - curr_q)
            self.q[s, a] = new_q

            s = sp

    def get_name(self):
        return "qlearning%s" % super(QLearningAgent, self).get_name()
