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
        
        self.prior_p= prior_p
        self.transition_p = transition_p
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
        
        # Step 1. Initialize variables
        
        #store probabilities of hidden state at each step 
        viterbi_table = np.zeros((len(decode_observation_states), len(self.hidden_states)))
        #store best path for traceback
        best_path = np.zeros(len(decode_observation_states))   
        
        # Initialize path (i.e., np.arrays) to store the hidden sequence states returning the maximum probability
        #path = np.zeros((len(decode_observation_states), len(hidden_s)))
        
        
        # Step 2. Calculate Probabilities
       
        # Calculate probabilities of each hidden state for the first observed state in the sequence
        first_obs_id = self.observation_states_dict[decode_observation_states[0]]
        viterbi_table[0, :] = self.prior_p * self.emission_p[:, first_obs_id]
       
        # Then for every subsequent observation state in the sequence, 
        # calculate the probability of every hidden state sequence, 
        # and select the hidden state sequence with the highest probability 
        for obs_state in range(1, len(decode_observation_states)):
           
            obs_id = self.observation_states_dict[decode_observation_states[obs_state]]
           
            for hidden_state in range(0, len(self.hidden_states)):
               
                trans_p = np.array([
                                    viterbi_table[obs_state - 1, prev_state] * self.transition_p[prev_state, obs_id] 
                                    for prev_state in range(len(self.hidden_states))
                                    ])

                best_trans_p = np.max(trans_p)
                viterbi_table[obs_state, hidden_state] = best_trans_p * self.emission_p[hidden_state, obs_id]

            
        # Step 3. Traceback 


        # Step 4. Return best hidden state sequence 
        #return best_hidden_state_path
        
