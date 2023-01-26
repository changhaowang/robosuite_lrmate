from rlkit.samplers.rollout_functions import rollout
from rlkit.torch.pytorch_util import set_gpu_mode
import argparse
import torch
import uuid
from rlkit.core import logger

import robosuite as suite
from robosuite.controllers import load_controller_config
from robosuite.utils.input_utils import *
from robosuite.wrappers import GymWrapper

filename = str(uuid.uuid4())


def simulate_policy(args):
    data = torch.load(args.file)
    policy = data['evaluation/policy']
    # env = data['evaluation/env']
    render_options = {}
    options = {}
    
    options["env_name"] = "Wipe"
    options["robots"] = "LRmate"
    
    controller_name = "OSC_POSE"

    # Load the desired controller
    options["controller_configs"] = suite.load_controller_config(default_controller=controller_name)
    env = GymWrapper(suite.make(
        **options,
        has_renderer=True,
        has_offscreen_renderer=False,
        ignore_done=True,
        use_camera_obs=False,
        control_freq=20,
    ))
    print("Policy loaded")
    if args.gpu:
        set_gpu_mode(True)
        policy.cuda()
    while True:
        path = rollout(
            env,
            policy,
            max_path_length=args.H,
            render=True,
        )
        if hasattr(env, "log_diagnostics"):
            env.log_diagnostics([path])
        logger.dump_tabular()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str,
                        help='path to the snapshot file')
    parser.add_argument('--H', type=int, default=300,
                        help='Max length of rollout')
    parser.add_argument('--gpu', action='store_true')
    args = parser.parse_args()

    simulate_policy(args)
