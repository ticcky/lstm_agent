"""
TicTacToe Reinforcement Learning
--------------------------------
Author: Lukas Zilka (lukas@zilka.me)

The learning algorithms in rl.py should be completely oblivious to what game
they are playing. This file `tictactoe.py` contains implementation of an
environment for the tic-tac-toe game, which can be used for teaching the agents
and playing against them.

Example of training the agents can be found in `train.sh`. This scripts trains
the agents using both algorithms (Q-learning and SARSA) to be competetive in
tic-tac-toe. Additionally it plots graphs into the plots/folder.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import deque
import seaborn as sbt
sbt.set()

from rl import (EnvironmentFeedback, SARSAAgent, QLearningAgent, HumanAgent,
                Agent)
from lstm_agent import LSTMAgent


# A set of winning positions. If some player has symbols on all of the places
# in one of the positions bellow he won.
WINNING_SETS = [
    [0, 1, 2],
    [3, 4, 5],
    [6, 7, 8],

    [0, 3, 6],
    [1, 4, 7],
    [2, 5, 8],

    [0, 4, 8],
    [6, 4, 2]
]

PLAYER1 = 1
PLAYER2 = -1


def other_player(player):
    if player == PLAYER1:
        return PLAYER2
    else:
        return PLAYER1


class RandomTicTacToeAgent(Agent):
    """
    Agent that plays randomly, but only the allowed moves.
    """
    def run(self):
        env_feedback = yield  # Initialization.
        while True:
            candidates = []
            for i, sc in enumerate(env_feedback.state):
                if sc == 0:
                    candidates.append(i)
            if candidates:
                a = np.random.choice(candidates)
            else:
                a = 0
            env_feedback = yield a

    def get_name(self):
        return "random"



class TicTacToeEnvironement:
    """
    Implementation of environment for tic-tac-toe game where two agents play
    against each other.
    """
    def __init__(self, agent1, agent2, agent1_states, agent2_states):
        self.agent1 = agent1
        self.agent2 = agent2
        self.agent1_states = agent1_states
        self.agent2_states = agent2_states

        self.board = self._prepare_clean_board()
        self.turn = PLAYER1
        self.state_map = {}
        self.total_reward = {PLAYER1: 0.0, PLAYER2: 0.0}
        self.reward_per_episode = {PLAYER1: [], PLAYER2: []}
        self.reward_per_episode100_plot = {PLAYER1: [], PLAYER2: []}
        self.reward_per_episode1000_plot = {PLAYER1: [], PLAYER2: []}
        self.reward_per_episode100 = {PLAYER1: deque(maxlen=100), PLAYER2: deque(maxlen=100)}
        self.episode = 0
        self.illegal_move_performed = False
        self.state_map_cache = {}
        self.last_actions = {PLAYER1: 9, PLAYER2: 9}

    def __unicode__(self):
        """Draw the representation of the board that can be printed to terminal.
        """
        res = []
        res.append('---\n')
        for y in range(3):
            for x in range(3):
                curr = self.board[y * 3 + x]
                if curr == PLAYER1:
                    res.append('o')
                elif curr == PLAYER2:
                    res.append('x')
                else:
                    res.append(' ')
            res.append('\n')
        res.append('---\n')
        if self.episode > 1:
            #res.append('P1: %.5f ' % self.reward_per_episode[PLAYER1][-1], )
            #res.append('P2: %.5f' % self.reward_per_episode[PLAYER2][-1], )
            res.append('P1(-1000): %.5f ' % (sum(self.reward_per_episode[PLAYER1][-1000:]) / 1000), )
            res.append('P2(-1000): %.5f' % (sum(self.reward_per_episode[PLAYER2][-1000:]) / 1000), )
        res.append('\n')

        return "".join(res)

    def new_episode(self):
        """Initialize new episode."""
        self.episode += 1
        self.board = self._prepare_clean_board()

        self.illegal_move = {PLAYER1: False, PLAYER2: False}

        self.a1_run = self.agent1.run()
        self.a1_run.next()

        self.a2_run = self.agent2.run()
        self.a2_run.next()

        self.turn = PLAYER1
        self.illegal_move_performed = None
        self.last_actions = {PLAYER1: 9, PLAYER2: 9}

    def _prepare_clean_board(self):
        return [0] * 9

    def plot_reward(self, filename):
        """Plot the reward per episode vs episode collected so far by the agents."""
        fig = plt.figure()
        axis = plt.gca()
        axis.set_ylim([-1.0, 1.0])

        #p1 = plt.plot(self.reward_per_episode[PLAYER1], label='Player 1 (%s)' % self.agent1.get_name())
        #p2 = plt.plot(self.reward_per_episode[PLAYER2], label='Player 2 (%s)' % self.agent2.get_name())

        p1x = plt.plot(self.reward_per_episode100_plot[PLAYER1], label='Player 1 (100 eps) (%s)' % self.agent1.get_name(), linewidth=1)
        p2x = plt.plot(self.reward_per_episode100_plot[PLAYER2], label='Player 2 (100 eps) (%s)' % self.agent2.get_name(), linewidth=1)

        p1xx = plt.plot(self.reward_per_episode1000_plot[PLAYER1], label='Player 1 (1000 eps) (%s)' % self.agent1.get_name(), linewidth=1)
        p2xx = plt.plot(self.reward_per_episode1000_plot[PLAYER2], label='Player 2 (1000 eps) (%s)' % self.agent2.get_name(), linewidth=1)
        legend = plt.legend(loc='upper right')

        plt.ylabel('Reward')
        plt.xlabel('Episode')
        plt.savefig(filename)
        plt.close()

    def get_state_standard(self):
        """Get a representation of the state of the game."""
        state = tuple(self.board)
        return state

    def state_to_int_standard(self, state):
        """Map the state to an int."""
        if state in self.state_map_cache:
            return self.state_map_cache[state]
        else:
            res = 0
            for i, sc in enumerate(state):
                res += (3 ** i) * (sc + 1)

            self.state_map_cache[state] = res
            return res

    def get_state_actions(self):
        return (self.last_actions[PLAYER1], self.last_actions[PLAYER2], self.board)

    def state_to_int_actions(self, state):
        return state[0] * 10 + state[1]

    def step(self):
        """Perform one step of the game.

        We define one step as one move (either player 1 draws symbol, or
        player 2 draw symbol).
        """
        if self.turn == PLAYER1:
            agent = self.a1_run
            get_state, state_to_int = self._get_state_fn(self.agent1_states)
        elif self.turn == PLAYER2:
            agent = self.a2_run
            get_state, state_to_int = self._get_state_fn(self.agent2_states)
        else:
            assert False, "Illegal state."

        s = get_state()

        env_feedback = EnvironmentFeedback(
            reward=0.0,
            state=s,
            state_id=state_to_int(s),
            final=False
        )
        a = agent.send(env_feedback)
        self._perform_action(a)

        self.turn = other_player(self.turn)

    def _get_state_fn(self, which):
        if which == 'standard':
            return self.get_state_standard, self.state_to_int_standard
        elif which == 'actions':
            return self.get_state_actions, self.state_to_int_actions
        else:
            assert False


    def finalize_episode(self):
        """Calculate and send the rewards to the players for the end of game."""
        r1, r2 = self._calculate_reward()

        get_state1, state_to_int1 = self._get_state_fn(self.agent1_states)
        get_state2, state_to_int2 = self._get_state_fn(self.agent2_states)
        s1 = get_state1()
        s2 = get_state2()

        self._send_feedback(PLAYER1, r1, s1, final=True)
        self._send_feedback(PLAYER2, r2, s2, final=True)

        self.total_reward[PLAYER1] += r1
        self.total_reward[PLAYER2] += r2
        self.reward_per_episode[PLAYER1].append(r1) #self.total_reward[PLAYER1] / self.episode)
        self.reward_per_episode[PLAYER2].append(r2) #self.total_reward[PLAYER2] / self.episode)
        self.reward_per_episode100_plot[PLAYER1].append(sum(self.reward_per_episode[PLAYER1][-100:]) / 100)
        self.reward_per_episode100_plot[PLAYER2].append(sum(self.reward_per_episode[PLAYER2][-100:]) / 100)
        self.reward_per_episode1000_plot[PLAYER1].append(sum(self.reward_per_episode[PLAYER1][-1000:]) / 1000)
        self.reward_per_episode1000_plot[PLAYER2].append(sum(self.reward_per_episode[PLAYER2][-1000:]) / 1000)
        #self.reward_per_episode100[PLAYER1].append(r1)
        #self.reward_per_episode100[PLAYER2].append(r2)

    def _calculate_reward(self):
        """Calculate rewards for the current state."""
        if self.has_won(PLAYER1) or self.illegal_move_performed == PLAYER2:
            r1 = 1.0
            r2 = -1.0
        elif self.has_won(PLAYER2) or self.illegal_move_performed == PLAYER1:
            r1 = -1.0
            r2 = 1.0
        else:
            r1 = 0.0
            r2 = 0.0

        return r1, r2

    def _send_feedback(self, player, r, s, final=False):
        """Send reward and current state to the players."""
        if player == PLAYER1:
            agent = self.a1_run
            get_state, state_to_int = self._get_state_fn(self.agent1_states)
        elif player == PLAYER2:
            agent = self.a2_run
            get_state, state_to_int = self._get_state_fn(self.agent2_states)
        else:
            raise Exception()

        s_id = state_to_int(s)

        env_feedback = EnvironmentFeedback(
            reward=r,
            state=s,
            state_id=s_id,
            final=final
        )
        agent.send(env_feedback)

    def _perform_action(self, a):
        """Perform action a on as the player whose turn it is."""
        if self.board[a] == 0:
            self.board[a] = self.turn
        else:
            self.illegal_move_performed = self.turn

        self.last_actions[self.turn] = a

        return 0.0

    def has_won(self, player):
        """Has the given player won?"""
        for winner in WINNING_SETS:
            if all(self.board[x] == player for x in winner):
                return True

        return False

    def is_board_full(self):
        """Can there be no more symbols placed on the board?"""
        return not (0 in self.board)

    def player_performed_illegal_move(self):
        """Has any player performed an illegal move?

        When a player performs an illegal move he loses the game.
        """
        return self.illegal_move_performed

    def is_finished(self):
        """Is the game finished?

        Returns true if either player already won, there's no more space on the
        board or one of the players performed an illegal move."""
        return (self.has_won(PLAYER1) or self.has_won(PLAYER2)
                or self.is_board_full() or self.player_performed_illegal_move())


class ActionTicTacToeEnvironment(TicTacToeEnvironement):
    def get_state(self):
        return (self.last_actions[PLAYER1], self.last_actions[PLAYER2], self.board)

    def state_to_int(self, state):
        return state[0] * 10 + state[1]


def main(episode_modulo, alpha, gamma, epsilon, agent, plot_reward,
         n_episodes, states):
    """
    Initialize and run a tic-tac-toe game.

    :param episode_modulo: Modulo factor for printing output and saving.
    :param alpha: Learning rate.
    :param gamma: Reward decay.
    :param epsilon: Exploration rate.
    :param agent1_type: Type of the first agent.
    :param agent2_type: Type of the second agent.
    :param agent1_load: Filename of the state of the first agent.
    :param agent1_save: Filename where to save state of the first agent.
    :param agent2_load: Filename of the state of the second agent.
    :param agent2_save: Filename where to save state of the second agent.
    :param plot_reward: Filename where to plot reward per episode.
    :param n_episodes: How many episodes to run. None means infinitely many.
    """

    n_actions = 9

    assert len(agent) == 2, 'Need to specify two agents.'

    agent1_vars = parse_agent_str(agent[0])
    agent2_vars = parse_agent_str(agent[1])
    a1, a1_states = get_agent_instance(agent1_vars, n_actions)
    a2, a2_states = get_agent_instance(agent2_vars, n_actions)

    agent1_save = agent1_vars.get('save')

    agent2_save = agent2_vars.get('save')

    def save_callback():
        if agent1_save:
            with open(agent1_save, 'w') as f_out:
                a1.save(f_out)

        if agent2_save:
            with open(agent2_save, 'w') as f_out:
                a2.save(f_out)

        if plot_reward:
            env.plot_reward(plot_reward)

    env = TicTacToeEnvironement(a1, a2, a1_states, a2_states)
    run_game(env, n_episodes, episode_modulo, save_callback)

def get_agent_instance(agent_vars, n_actions):
    """
    Create instance and load state of the agent.

    :return: Instance of an agent.
    """
    agent_type = agent_vars['type']
    load = agent_vars.get('load')
    alpha = float(agent_vars.get('a', 0.1))
    gamma = float(agent_vars.get('g', 0.9))
    epsilon = float(agent_vars.get('e', 0.1))
    states = agent_vars.get('states', 'standard')

    if states == 'standard':
        n_states = 3**9
    elif states == 'actions':
        n_states = 10 * 10
    else:
        assert False


    if agent_type == 'sarsa':
        res = SARSAAgent(alpha, gamma, n_actions, n_states, epsilon)
        if load:
            res.load(load)

        return res, states
    elif agent_type == 'qlearning':
        res = QLearningAgent(alpha, gamma, n_actions, n_states, epsilon)
        if load:
            res.load(load)

        return res, states
    elif agent_type == 'lstm':
        res = LSTMAgent(alpha, gamma, n_actions, n_states, epsilon)
        if load:
            res.load(open(load))

        return res, states
    elif agent_type == 'human':
        return HumanAgent(), states
    elif agent_type == 'random':
        return RandomTicTacToeAgent(), states
    else:
        print 'Unknown agent type:', agent_type
        return exit(1)


def parse_agent_str(agent):
    res = {}
    for arg in agent.split(','):
        key, value = arg.split('=', 1)
        res[key] = value

    return res


def run_game(env, n_episodes, episode_modulo, modulo_callback):
    """
    Run tic-tac-toe game.

    :param env: Environment.
    :param n_episodes: How many episodes to run.
    :param episode_modulo: Print sample game each episode when the episode_modulo
                           equality is true: episode % episode_modulo == 0
    :param modulo_callback: Function to call for saving when the episode_modulo
                            equality is true.
    """
    while n_episodes is None or env.episode < n_episodes:
        if env.episode % episode_modulo == 0:
            print '#' * 100
            print '# episode', env.episode

        env.new_episode()
        while not env.is_finished():
            env.step()
            if env.episode % episode_modulo == 0:
                print(unicode(env))

        env.finalize_episode()

        if env.episode % episode_modulo == 0:
            modulo_callback()


if __name__ == '__main__':
    import utils
    utils.pdb_on_error()

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--epsilon', type=float, default=0.1)
    parser.add_argument('--episode_modulo', type=float, default=1000)
    parser.add_argument('--agent', action='append')
    #parser.add_argument('--agent1_type', default='sarsa')
    #parser.add_argument('--agent2_type', default='sarsa')
    #parser.add_argument('--agent1_load', default=None)
    #parser.add_argument('--agent1_save', default=None)
    #parser.add_argument('--agent2_load', default=None)
    #parser.add_argument('--agent2_save', default=None)
    parser.add_argument('--plot_reward', default=None)
    parser.add_argument('--n_episodes', type=int, default=None)
    parser.add_argument('--states', default='standard')

    args = parser.parse_args()

    main(**vars(args))