"""
Whether you are loading from a partial state_dict, which is missing some keys, 
or loading a state_dict with more keys than the model that you are loading into, 
you can set the strict argument to False in the load_state_dict() function to ignore non-matching keys. 
In this recipe, we will experiment with warmstarting a model using parameters of a different model.
"""
