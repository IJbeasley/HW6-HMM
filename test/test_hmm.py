import pytest
from hmm import HiddenMarkovModel
import numpy as np




def test_mini_weather():
    """
    TODO: 
    Create an instance of your HMM class using the "small_weather_hmm.npz" file. 
    Run the Forward and Viterbi algorithms on the observation sequence in the "small_weather_input_output.npz" file.

    Ensure that the output of your Forward algorithm is correct. 

    Ensure that the output of your Viterbi algorithm correct. 
    Assert that the state sequence returned is in the right order, has the right number of states, etc. 

    In addition, check for at least 2 edge cases using this toy model. 
    """

    mini_hmm=np.load('./data/mini_weather_hmm.npz')
    mini_input=np.load('./data/mini_weather_sequences.npz')
    
    mini_hmm = HiddenMarkovModel(
                                 observation_states = mini_hmm['observation_states'],
                                 hidden_states = mini_hmm['hidden_states'],
                                 prior_p = mini_hmm['prior_p'],
                                 transition_p = mini_hmm['transition_p'],
                                 emission_p = mini_hmm['emission_p']
                                )
                                
    # Ensure that the output of your Forward algorithm is correct.  
    forward_prob = mini_hmm.forward(mini_input['observation_state_sequence'])
    assert forward_prob > 0, "The output of the forward algorithm for the mini dataset is not correct"
    # print(forward_prob)
    # raise ValueError("hfhgh")
    # assert forward_prob
    
    
    # Ensure that the output of your Viterbi algorithm is correct
    viterbi = mini_hmm.viterbi(mini_input['observation_state_sequence'])
    assert len(viterbi) == len(mini_input['best_hidden_state_sequence']),  "The calculated viterbi best hidden state sequence is incorrect for mini weather dataset"
    assert np.all(viterbi == mini_input['best_hidden_state_sequence']), "The calculated viterbi best hidden state sequence is incorrect for mini weather dataset"




def test_full_weather():

    """
    TODO: 
    Create an instance of your HMM class using the "full_weather_hmm.npz" file. 
    Run the Forward and Viterbi algorithms on the observation sequence in the "full_weather_input_output.npz" file
        
    Ensure that the output of your Viterbi algorithm correct. 
    Assert that the state sequence returned is in the right order, has the right number of states, etc. 

    """
    
    full_hmm=np.load('./data/full_weather_hmm.npz')
    full_input=np.load('./data/full_weather_sequences.npz')
    
    full_hmm = HiddenMarkovModel(
                                 observation_states = full_hmm['observation_states'],
                                 hidden_states = full_hmm['hidden_states'],
                                 prior_p = full_hmm['prior_p'],
                                 transition_p = full_hmm['transition_p'],
                                 emission_p = full_hmm['emission_p']
                                )
    
    # Ensure that the output of your Forward algorithm is correct.  
    forward_prob = full_hmm.forward(full_input['observation_state_sequence'])   
    assert np.all(forward_prob > 0), "The output of the forward algorithm for the full datasets is not correct"
                                
    # Ensure that the output of your Viterbi algorithm is correct                            
    viterbi = full_hmm.viterbi(full_input['observation_state_sequence'])
    
    assert len(viterbi) == len(full_input['best_hidden_state_sequence']),  "The calculated viterbi best hidden state sequence is incorrect for full weather dataset"
    assert np.all(viterbi == full_input['best_hidden_state_sequence']), "The calculated viterbi best hidden state sequence is incorrect for full weather dataset"



# Unit test to check that the HiddenMarkovModel fails correctly when negative prior probs are provided
def test_negative_prior_p():
  
    mini_hmm=np.load('./data/mini_weather_hmm.npz')
    
    try:
        mini_hmm = HiddenMarkovModel(
                                     observation_states = mini_hmm['observation_states'],
                                     hidden_states = mini_hmm['hidden_states'],
                                     prior_p = -1 * mini_hmm['prior_p'], #multiply by negative -1 to make all prior probs negative
                                     transition_p = mini_hmm['transition_p'],
                                     emission_p = mini_hmm['emission_p']
                                )
       
        assert False, "HiddenMarkovModel should have failed on negative prior probabilities"
       
    except ValueError as e:
       print(str(e))
       assert str(e) == "Prior probabilities cannot be negative", "Negative prior probabilities should have raised a different ValueError in vertbi function"
   
   

# Unit test to check that the viterbi alg fails correctly when invalid states in decode_observation_states are provided
def test_invalid_decode_obs_seq():
  
    mini_hmm=np.load('./data/mini_weather_hmm.npz')
    
    mini_hmm = HiddenMarkovModel(
                                 observation_states = mini_hmm['observation_states'],
                                 hidden_states = mini_hmm['hidden_states'],
                                 prior_p = mini_hmm['prior_p'],
                                 transition_p = mini_hmm['transition_p'],
                                 emission_p = mini_hmm['emission_p']
                                )
                                
    mini_invalid_input = ['dog', 'wolf', 'dog', 'cat', 'wolf']
    
    try:
       
       mini_hmm.viterbi(mini_invalid_input)
       assert False, "Viterbi algorithm should have failed on invalid decode_observation_states"
       
    except ValueError as e:
       assert str(e) == "Invalid observation state: dog", "Invalid decode_observation_states should have raised a different ValueError in vertbi function"
   
   
  
  










