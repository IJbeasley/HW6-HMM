import numpy as np
class HiddenMarkovModel:
    """
    Class for Hidden Markov Model 
    """

    def __init__(self, observation_states: np.ndarray, hidden_states: np.ndarray, prior_p: np.ndarray, transition_p: np.ndarray, emission_p: np.ndarray):
        """

        Initialization of HMM object

        Args:
            observation_states (np.ndarray): observed states 
            hidden_states (np.ndarray): hidden states 
            prior_p (np.ndarray): prior probabities of hidden states 
            transition_p (np.ndarray): transition probabilites between hidden states
            emission_p (np.ndarray): emission probabilites from transition to hidden states 
        """             
        
        self.observation_states = observation_states
        self.observation_states_dict = {state: index for index, state in enumerate(list(self.observation_states))}

        self.hidden_states = hidden_states
        self.hidden_states_dict = {index: state for index, state in enumerate(list(self.hidden_states))}
        
        # Check probability input 
        # prior_p sums to 1
        if not np.isclose(np.sum(prior_p),1):
           raise ValueError("Prior probabilities need to sum to 1")
         
        # prior_p should be 1d
        if len(prior_p.shape) != 1:
           raise ValueError("Prior probability array should be 1D")
        
        # no negative probilities in prior_p
        if np.any(prior_p < 0):
           raise ValueError("Prior probabilities cannot be negative")
        
        # number of prior probabilities is equal to the number of hidden states
        if len(prior_p) != len(hidden_states):
           raise ValueError("The number of prior probabilities should correspond to the number of hidden states")
         
        self.prior_p = prior_p
        
        # Each row of transition_p sums to 1
        if np.any(np.sum(transition_p, axis = 1) != 1):
           raise ValueError("Every row of transition probability matrix must sum to 1")
        
        
        # no negative probilities in transition_p
        if np.any(transition_p < 0):
           raise ValueError("Transition probabilities cannot be negative")
         
        # transition_p should be square
        if transition_p.shape[0] != transition_p.shape[1]:
           raise ValueError("Transition probability matrix should be square")
        
        # transition_p should be 2d
        if len(transition_p.shape) != 2:
           raise ValueError("Transition probability matrix should be 2D")
         
        # transition_p should have the same number of states as hidden states
        if transition_p.shape[0] != len(hidden_states):
           raise ValueError("The number of states in the transition probability matrix should be equal to the number of hidden states")
        
        self.transition_p = transition_p
        
        # Each row of emission_p sums to 1
        if np.any(np.sum(emission_p, axis = 1) != 1):
           raise ValueError("Every row of emission probability matrix must sum to 1")
        
        # emission probabilities should be non-negative
        if np.any(emission_p < 0):
           raise ValueError("Emission probabilities cannot be negative") 
         
        if len(emission_p) != len(hidden_states):
           raise ValueError("The number of emission probabilities should correspond to number of hidden states")
         
        if len(emission_p.shape) != 2:
           raise ValueError("Emission probabilities array should be 1D")
         
        self.emission_p = emission_p


    def forward(self, input_observation_states: np.ndarray) -> float:
        """
        TODO 

        This function runs the forward algorithm on an input sequence of observation states

        Args:
            input_observation_states (np.ndarray): observation sequence to run forward algorithm on 

        Returns:
            forward_probability (float): forward probability (likelihood) for the input observed sequence  
        """   
        
        # Step 0 - check inputs
        
        # Fail if input state sequence is empty 
        if len(input_observation_states) == 0:
           raise ValueError("Input state sequence shouldn't be empty")
        
        # Fail if input state sequence includes states not in our observed states dictionary
        for state in input_observation_states:
            if state not in self.observation_states_dict:
                raise ValueError(f"Invalid observation state: {state}")
         
        # Step 1. Initialize variables 
        # alpha (Forward Probability Table)
        alpha = np.zeros((len(input_observation_states), len(self.hidden_states)))
        
        # Step 2. Calculate probabilities
        
        # Calculate probabilities of each hidden state for the first observed state in the sequence
        first_obs_idx = self.observation_states_dict[input_observation_states[0]]
        alpha[0, :] = self.prior_p * self.emission_p[:, first_obs_idx]
        
        # Compute probabilities for subsequent observed states in the sequence (i.e. time steps)
        for t in range(1, len(input_observation_states)):
            obs_idx = self.observation_states_dict[input_observation_states[t]]
        
            for curr_state in range(len(self.hidden_states)):
                # Compute probability by summing over all previous states
                alpha[t, curr_state] = np.sum(alpha[t - 1, :] * self.transition_p[:, curr_state]) * self.emission_p[curr_state, obs_idx]

        
        # Step 3. Return final probability 
        
        forward_probability = np.sum(alpha[-1, :])
        
        return forward_probability

        


    def viterbi(self, decode_observation_states: np.ndarray) -> list:
        """
        TODO

        This function runs the viterbi algorithm on an input sequence of observation states

        Args:
            decode_observation_states (np.ndarray): observation state sequence to decode 

        Returns:
            best_hidden_state_sequence(list): most likely list of hidden states that generated the sequence observed states
        """        
        # Step 0 - check inputs
        
        # Fail if decode state sequence is empty 
        if len(decode_observation_states) == 0:
           raise ValueError("Decode state sequence shouldn't be empty")
        
        # Fail if decode state sequence includes states not in our observed states dictionary
        for state in decode_observation_states:
            if state not in self.observation_states_dict:
                raise ValueError(f"Invalid observation state: {state}")
        
        
        # Step 1. Initialize variables
        
        #store probabilities of hidden state at each step 
        viterbi_table = np.zeros((len(decode_observation_states), len(self.hidden_states)))
        #store best path for traceback
        best_path = np.zeros(len(decode_observation_states), dtype=int)   
        
        # Store the best previous state at each time step for traceback
        backpointer =  np.zeros((len(decode_observation_states), len(self.hidden_states)), dtype=int)
        
        # Step 2. Calculate Probabilities
       
        # Calculate probabilities of each hidden state for the first observed state in the sequence
        first_obs_id = self.observation_states_dict[decode_observation_states[0]]
        viterbi_table[0, :] = self.prior_p * self.emission_p[:, first_obs_id]
       
        # Then for every subsequent observation state in the sequence (i.e. time steps)
        # calculate the probability of every hidden state sequence, 
        # and select the hidden state sequence with the highest probability 
        for t in range(1, len(decode_observation_states)):
           
            obs_idx = self.observation_states_dict[decode_observation_states[t]]
           
            for curr_state in range(0, len(self.hidden_states)):
                # # Compute transition probabilities from all previous states
                # trans_p = viterbi_table[t - 1, :] * self.transition_p[:, hs]
                
                            # Calculate probabilities for all possible previous states
                probs = (viterbi_table[t-1, :] * 
                         self.transition_p[:, curr_state] * 
                         self.emission_p[curr_state, obs_idx])
            
                # Store maximum probability 
                viterbi_table[t, curr_state] = np.max(probs)
                # best previous state
                backpointer[t, curr_state] = np.argmax(probs)
                
                # # Select the most probable previous state
                # best_prev_state = np.argmax(trans_p)
                # 
                # # Store best probability and path
                # viterbi_table[t, hs] = trans_p[best_prev_state] * self.emission_p[hs, obs_idx]
                # backpointer[t, hs] = best_prev_state                
               
                # trans_p = np.array([
                #                     viterbi_table[t - 1, prev_state] * self.transition_p[prev_state, hs] 
                #                     for prev_state in range(len(self.hidden_states))
                #                     ])
                # 
                # best_trans_p = np.max(trans_p)
                # viterbi_table[t, hs] = best_trans_p * self.emission_p[hs, obs_idx]

            
        # Step 3. Traceback 
        
        # Find most probable final state
        best_path[-1] = np.argmax(viterbi_table[-1, :])
        
        # Iterate backward to reconstruct the best path
        for t in range(len(decode_observation_states) - 1, 0, -1):
            best_path[t - 1] = backpointer[t, best_path[t]]

        # Convert best path indices to hidden state labels
        best_hidden_state_path = [self.hidden_states[i] for i in best_path]

        # Step 4. Return best hidden state sequence 
        return best_hidden_state_path
        
