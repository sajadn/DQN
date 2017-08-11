import argparse
from . import config

parser = argparse.ArgumentParser()
parser.add_argument("envType", type=str)
parser.add_argument("envName", type=str)
parser.add_argument("action", type=str)
parser.add_argument("-lr","--learning_rate", type=float, default=0.00025)
parser.add_argument("-e","--epsilon", type = float, default = 1)
parser.add_argument("-epsilon_end", type=float, default=0.1)
parser.add_argument("-el","--epsilon_last", type=float, default=0.01)
parser.add_argument("-efr","--ep_first_reduction", type=int, default=10**6)
parser.add_argument("-ep_second_reduction", type=int, default=24*10**4)
parser.add_argument("-y", type=float, default=0.99)
parser.add_argument("-size_of_experience", type=int, default=10**6)
parser.add_argument("-mbs", "--mini_batch_size", type=int, default=32)
parser.add_argument("-tu","--target_update", type=int, default=10**4)
parser.add_argument("-rf","--regularization_factor", type =float, default= 0.001)
parser.add_argument("-ies","--initial_experience_sizes", type= int , default=10**6)
parser.add_argument("-sfs","--stacked_frame_size", type=int, default=4)
parser.add_argument("-b","--beta", type = float, default=0.4)
parser.add_argument("-a","--alpha", type=float, default=0.06)
parser.add_argument("-ep","--epsilon_prioritized", type=float, default= 0.001)
parser.add_argument("-ec","--error_clip", type=int, default=1)
parser.add_argument("-fs","--frame_skipping", type=int, default=4)
parser.add_argument("-tf","--train_frequency", type=int, default=1)
parser.add_argument("-mx","--max_step", type=int, default=2000000)
parser.add_argument("-r","--regularization", type=int, default=0)
parser.add_argument("-rn","--remove_no_op", type=int, default=0)
parser.add_argument("-dd","--doubleDQN", type=int, default=0)
parser.add_argument("-pe","--prioritized_experience", type=int, default=0)
parser.add_argument("-gf","--GPU_fraction", type=float, default=1)
parser.add_argument("-gn","--GPU_number", type=int, default=0)
config.params = parser.parse_args()
names = ["envType", "envName", "learning_rate","epsilon","target_update",
         "regularization_factor","stacked_frame_size", "error_clip",
          "frame_skipping", "train_frequency", "regularization",
           "remove_no_op", "doubleDQN", "prioritized_experience"]

print (config.params)
folder_name = ""
for name in names:
    folder_name += name+str(getattr(config.params, name))+"_"
config.params.folder_name = folder_name

from .agents.agent import run

run()
