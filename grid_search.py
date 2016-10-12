import itertools
import os
from collections import OrderedDict

def get_params():
	params = OrderedDict([
					( 'learning_rate' , ['.001','.0005']),
					('embedding_dim' , ['32','64','128']),
					('dropout_factor' , ['.5','1.0']),
					('num_filters' , ['32','64','128']),
					('filter_sizes' , ['2,3,4,5,6','3,4,5','4,5,6']),
					('fc_layers', ['1','2']),
					('l2_reg_lambda', ['0','1','3']),
					('activation_func', ['relu','tanh'])
	 				])
	return params

def run_search():
	params = get_params()
	hp_values = itertools.product(*params.values())
	hyper_params = params.keys()

	for hp_val in hp_values:
		command = ('python3.5 train.py ' + 
			' '.join('--{} {}'.format(arg,val) for arg,val in zip(hyper_params,hp_val))
			)
		print(command)
		os.system(command)

if __name__ == "__main__":
    run_search()
