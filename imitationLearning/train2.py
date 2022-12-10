import argparse
import time
import datetime
import torch_ac
import tensorboardX
import sys
from torch.utils.data import DataLoader
from utils.dataloader import LoadData
import numpy as np
from torch.optim import Adam

import utils
from utils import device
from torch.utils.data.sampler import SubsetRandomSampler

random_seed = 42
validation_split = .2
# Parse arguments
dataset_dir = './expert_vis15_il_vis15.pt'
parser = argparse.ArgumentParser()

# General parameters
parser.add_argument("--algo", required=True,
					help="algorithm to use: a2c | ppo (REQUIRED)")
parser.add_argument("--env", required=True,
					help="name of the environment to train on (REQUIRED)")
parser.add_argument("--model", default=None,
					help="name of the model (default: {ENV}_{ALGO}_{TIME})")
parser.add_argument("--seed", type=int, default=1,
					help="random seed (default: 1)")
parser.add_argument("--procs", type=int, default=16,
					help="number of processes (default: 16)")
parser.add_argument("--frames", type=int, default=10**7,
					help="number of frames of training (default: 1e7)")

# Parameters for main algorithm
parser.add_argument("--epochs", type=int, default=4,
					help="number of epochs for PPO (default: 4)")
parser.add_argument("--visibility", type=int, default=15,
					help="Number of visibility (default: 15)")
parser.add_argument("--lr", type=float, default=0.001,
                    help="learning rate (default: 0.001)")
parser.add_argument("--batch-size", type=int, default=256,
					help="batch size for PPO (default: 256)")
parser.add_argument("--value-loss-coef", type=float, default=0.5,
					help="value loss term coefficient (default: 0.5)")
parser.add_argument("--recurrence", type=int, default=1,
					help="number of time-steps gradient is backpropagated (default: 1). If > 1, a LSTM is added to the model to have memory.")
parser.add_argument("--text", action="store_true", default=False,
					help="add a GRU to the model to handle text input")

if __name__ == "__main__":
	args = parser.parse_args()

	args.mem = args.recurrence > 1

	# Set run dir

	date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
	args.env = 'MiniGrid-FourRooms-v0' 
	default_model_name = f"{args.env}_{args.algo}_seed{args.seed}_{date}"

	model_name = args.model or default_model_name
	model_dir = utils.get_model_dir(model_name)


	# Load loggers and Tensorboard writer

	txt_logger = utils.get_txt_logger(model_dir)
	csv_file, csv_logger = utils.get_csv_logger(model_dir)
	tb_writer = tensorboardX.SummaryWriter(model_dir)

	# Log command and all script arguments

	txt_logger.info("{}\n".format(" ".join(sys.argv)))
	txt_logger.info("{}\n".format(args))

	# Set seed for all randomness sources

	utils.seed(args.seed)

	# Set device

	txt_logger.info(f"Device: {device}\n")

	# Load training status

	try:
		status = utils.get_status(model_dir)
	except OSError:
		status = {"num_frames": 0, "update": 0}
	txt_logger.info("Training status loaded\n")

	# Load environments

	envs = []
	for i in range(args.procs):
		envs.append(utils.make_env(args.env, args.seed + 10000 * i, agent_view_size_param = args.visibility))
	txt_logger.info("Environments loaded\n")



	# Load observations preprocessor

	# Load model
	obs_space, preprocess_obss = utils.get_obss_preprocessor(envs[0].observation_space)
	if "vocab" in status:
		preprocess_obss.vocab.load_vocab(status["vocab"])
	txt_logger.info("Observations preprocessor loaded")

	agent = utils.Agent(envs[0].observation_space, envs[0].action_space, model_dir, num_envs=args.procs,
							use_memory=args.mem, use_text=args.text, train=True)

	agent.acmodel.to(device) 

	txt_logger.info("Model loaded\n")
	# txt_logger.info("{}\n".format(acmodel))


	optimizer = Adam(params=agent.acmodel.parameters(), lr=args.lr)
	# Load algo
	txt_logger.info("Optimizer loaded\n")

	# Train model

	num_frames = status["num_frames"]
	update = status["update"]
	start_time = time.time()

	#train data  and validation data 
	dataset = LoadData(dataset_dir)
	dataset_size = len(dataset)
	indices = list(range(dataset_size))
	split = int(np.floor(validation_split * dataset_size))

	if shuffle_dataset :
		np.random.seed(random_seed)
		np.random.shuffle(indices)
	train_indices, val_indices = indices[split:], indices[:split]

	# Creating PT data samplers and loaders:
	train_sampler = SubsetRandomSampler(train_indices)
	valid_sampler = SubsetRandomSampler(val_indices)

	train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch-size, shuffle=True,sampler=train_sampler,
						   num_workers=1,pin_memory=True, drop_last=True)

	validation_loader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=True,sampler=valid_sampler,
						   num_workers=1,pin_memory=True, drop_last=False)
	# Train model

	num_frames = status["num_frames"]
	update = status["update"]
	start_time = time.time()

	crss_ent = nn.CrossEntropyLoss()

	while epoch < args.frames:
		torch.cuda.empty_cache()
		train_iter = iter(train_loader)
		train_loss = 0
		for i in range(len(train_loader)):
			optimizer.zero_grad()
			state, action = next(train_iter)
			state = state.to(device, non_blocking=True)
			action = action.to(device, non_blocking=True)
			action_op = agent.action_prob(state)
			loss = crss_ent(action_op,action)

			loss.backward()
			optimizer.step()
			
			if i == 0:
				# Save the model that corresponds to the current epoch

				agent.acmodel.eval().cpu()
				torch.save(agent.acmodel.state_dict(), "models/imitation" + "_epoch_" + str(epoch) + ".pth")
				agent.acmodel.to(device).train()

				# Evaluate the model

				total_loss = 0 

				agent.eval()
				with torch.no_grad():

					val_iter = iter(validation_loader)
					for j in range(len(validation_loader)):

						state, action = next(val_iter)
						state = state.to(device, non_blocking=True)
						action = action.to(device, non_blocking=True)
						enhanced = generator(x)

						total_loss += crss_ent(action_op,action).item()
				total_loss = total_loss / len(validation_loader)
				print("Epoch %d, loss: %.4f" % (epoch,total_loss))
				agent.acmodel.train()
