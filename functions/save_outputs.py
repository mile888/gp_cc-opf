#!/usr/bin/env python


import numpy as np
import json


# Outputs
def to_dict(x, s, u, alpha, trace_covar):
        """ Store output data in a dictionary """

        IEEE_dict = {}
        IEEE_dict['x'] = x.tolist()
        IEEE_dict['s'] = s.tolist()
        IEEE_dict['u'] = u.tolist()
        IEEE_dict['alpha'] = alpha.tolist()
        IEEE_dict['trace_covar'] = trace_covar.tolist()
        
        return IEEE_dict
    
    
# Inputs
def to_dict_in(xd, xd_std, u, alpha):
        """ Store output data in a dictionary """

        IEEE_dict = {}
        IEEE_dict['xd'] = xd.tolist()
        IEEE_dict['xd_std'] = xd_std.tolist()
        IEEE_dict['u'] = u.tolist()
        IEEE_dict['alpha'] = alpha.tolist()
        
        return IEEE_dict
    

#Save and Load
def save_model(IEEE_dict, filename):
        """ Save model to a json file"""
        with open(filename + ".json", "w") as outfile:
            json.dump(IEEE_dict, outfile)


    
def load_model(filename):
        """ Create a new model from file"""
        with open(filename + ".json") as json_data:
            input_dict = json.load(json_data)
        return input_dict







