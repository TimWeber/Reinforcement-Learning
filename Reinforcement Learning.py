import numpy as np 

# Build Environment

height	=	6
width	=	8
n_actions = 4
n_episodes = 10000
epsilon_current = 1
epsilon_start = 1.
epsilon_decay = 0.01
epsilon_min = 0.8
learning_rate = 0.1
discount_factor = 0.98

def build_treasure_island(height, width):
	environment =	np.zeros([height*width])
	environment[33]	=	2
	environment[7]	=	3

	grass	=	np.array([9,10,	13,	16,	19,	22,	25,	26,	27,	28,	32,	34,	38,	39,	40])

	for g in range(len(grass)):
		environment[(grass[g])]	=		1

	#print(environment)
	print('Tresure Island \n 0: Seas, 1: Grass 2: Small Gold 3: Big Gold \n',
		np.transpose(np.reshape(environment,	(width,	height))))

	return environment

environment = build_treasure_island(height,	width)

def step(state, action, environment, height):
	global reward
	
	if action == 0:
		new_state	=	state - 1
	elif action == 1:
		new_state = state + 1
	elif action == 2:
		new_state = state + height
	elif action == 3:
		new_state = state - height

	# Reward function

	if environment[new_state] == 0:
		reward = -10
		end_episode = True

	elif environment[new_state] == 1:
		reward = -2
		end_episode = False

	elif environment[new_state]	== 2:
		reward == 30
		end_episode = True

	elif environment[new_state] == 3:
		reward = 90
		end_episode = True

	return new_state, reward, end_episode

# Build agent

Q_table = np.zeros([height*width, n_actions])

def epsilon_greedy_policy(state, Q_table, epsilon_current, epsilon_start, epsilon_decay, epsilon_min, n_actions):

	# Calculate Epsilon

	epsilon = epsilon_start - epsilon_current*epsilon_decay
	if epsilon_start < epsilon_min:
		epsilon = epsilon_min

	random_number = np.random.rand(1)

	if random_number > epsilon: # act greedy

		multiple_greedy_actions = []
		greedy_action_value = np.max(Q_table[state])
		for a in range(n_actions):
			if Q_table[state, a] ==  greedy_action_value:
				multiple_greedy_actions.append(a)

		action = multiple_greedy_actions[np.random.randint(len(multiple_greedy_actions))]

	else:
		action = np.random.randint(4)

	return action

# Train agent

for e in range(n_episodes):
	start_state = 10
	state = start_state
	end_episode = False
	t = 0
	while not end_episode:
		action = epsilon_greedy_policy(state, Q_table, e, epsilon_start, epsilon_decay, epsilon_min, n_actions)
		prediction = Q_table[state, action]
		new_state, reward, end_episode = step(state, action, environment, height)
		target = reward + discount_factor*np.max(Q_table[new_state])
		Q_table[state, action] = Q_table[state, action] + learning_rate*(target - prediction)

		t += 1

		state = new_state

print('					action')
print('\n State 	N S E W 	Greedy Policy')
counter = -1
action_dict = {0:'N', 1:'S', 2:'E', 3:'W'}
for a in range(height*width):
	counter += 1

	if environment[counter] == 1:
		best_action = np.argmax(Q_table[counter])
		best_action = action_dict[best_action]
		print(counter, '    ', np.round(Q_table[counter]), '     ', best_action)

