# ECE209RL-FourRoom

there is a submodule named rl-starter-files that are used to train RL models and Imitation lerning models. 

When cloning this repository, please do a reursive clone like following:
``` bash
git clone --recursive git@github.com:luciochen233/ECE209RL-FourRoom.git

git submodule foreach -q --recursive 'git checkout $(git config -f $toplevel/.gitmodules submodule.$name.branch || echo master)'

git config --global status.submoduleSummary true

## this part of the tutorial is from Git submodules best practices

## https://gist.github.com/slavafomin/08670ec0c0e75b500edbaa5d43a5c93c
```
# Overview on how to genereate RL expert, expert training data, and imitation learning

## First, to generate the RL base line

Inside the rl-starter-files repo, we used the following command to train the RL baseline.
``` bash
python -m scripts.train --algo a2c --env MiniGrid-FourRooms-v0 --model FourRoom9 --save-interval 10 --frames 40000000 --recurrence 128 --batch-size 8192 --epochs 10 --frames-per-proc 512
## note that is will train the model using a2c for 40 million frames.

```

## Second, to generate the RL experts

In there, we have added visibility as a parameter so we can train experts with various visibilities

``` bash
python -m scripts.train --algo a2c --env MiniGrid-FourRooms-v0 --save-interval 10 --frames 40000000 --recurrence 128 --batch-size 8192 --epochs 10 --frames-per-proc 512 --visibility 9 --model FourRoomVisibility9

python -m scripts.train --algo a2c --env MiniGrid-FourRooms-v0 --save-interval 10 --frames 40000000 --recurrence 128 --batch-size 8192 --epochs 10 --frames-per-proc 512 --visibility 11 --model FourRoomVisibility11

python -m scripts.train --algo a2c --env MiniGrid-FourRooms-v0 --save-interval 10 --frames 40000000 --recurrence 128 --batch-size 8192 --epochs 10 --frames-per-proc 512 --visibility 13 --model FourRoomVisibility13

python -m scripts.train --algo a2c --env MiniGrid-FourRooms-v0 --save-interval 10 --frames 40000000 --recurrence 128 --batch-size 8192 --epochs 10 --frames-per-proc 512 --visibility 15 --model FourRoomVisibility15

```

## Third, generate Expert knowledge that can be used by imitation learning
``` bash
python -m scripts.visualize --env MiniGrid-FourRooms-v0 --model FourRoomVisibility15 --visibility 15 --memory --il_visibility 7 --save --episodes 30000
# expect run time is about 30 minutes for 30000 trajectories

```

## Fourth, copy the expert knowledge into the imitation learning folder
``` bash
cp ./rl-starter-files/expert_vis15_il_vis15.pt ./imitationLearning
```

## Fifth, run imitation learning on the expert knowledge (remember to go to the imitation learning folder)

refer to the imitation learning note, but I will paste that here as well

# Imitation learning modification

This contains a modification of the RL Starter Files that run imitation learning (bahvior cloning) with expert data.

to run the imitation learning, first generate the expert data using the other repo.
Once the expert data has benn obtained, run the following script to generate the imitation learning policy.


``` bash
python3 -m scripts.train2 --algo ppo --model Imitation --frames 20 --env MiniGrid-FourRooms-v0 --visibility 7 --recurrence 256 --dataset_dir "./expert_vis15_il_vis15.pt" --procs 256 --savename "vis15"

```
This will generate policy that are compatible with the rl-starter-file. It will be saved under the 
```imitationLearning/models ```
folder

## Finally, we can run the imitation models in rl-starter-files
create a named folder under rl-starter-files/storage
copy the imitation expert *.pt file into this named folder that you created
change the name to status.pt
run the visualization by using the following command:

``` bash
python -m scripts.visualize --env MiniGrid-FourRooms-v0 --model imitation --visibility 7 --memory

```

## to evaluate the performance, run the following command under rl-start-files

``` bash
python -m scripts.evaluate --env MiniGrid-FourRooms-v0 --model imitation --visibility 7 --memory
```


#

## Create dataset for imitation learning
```
python il_dataset.py --num_eps NUM_EPISODES --dir DATA_DIR [--render]
```
