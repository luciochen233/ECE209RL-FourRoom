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

Inside the rl-starter-files repo, we used the following command to train the RL baseline.
``` bash
python -m scripts.train --algo a2c --env MiniGrid-FourRooms-v0 --model FourRoom9 --save-interval 10 --frames 40000000 --recurrence 128 --batch-size 8192 --epochs 10 --frames-per-proc 512
## note that is will train the model using a2c for 40 million frames.

```

In there, we have added visibility as a parameter so we can train experts with various visibilities

``` bash
python -m scripts.train --algo a2c --env MiniGrid-FourRooms-v0 --save-interval 10 --frames 40000000 --recurrence 128 --batch-size 8192 --epochs 10 --frames-per-proc 512 --visibility 9 --model FourRoomVisibility9

python -m scripts.train --algo a2c --env MiniGrid-FourRooms-v0 --save-interval 10 --frames 40000000 --recurrence 128 --batch-size 8192 --epochs 10 --frames-per-proc 512 --visibility 11 --model FourRoomVisibility11

python -m scripts.train --algo a2c --env MiniGrid-FourRooms-v0 --save-interval 10 --frames 40000000 --recurrence 128 --batch-size 8192 --epochs 10 --frames-per-proc 512 --visibility 13 --model FourRoomVisibility13

python -m scripts.train --algo a2c --env MiniGrid-FourRooms-v0 --save-interval 10 --frames 40000000 --recurrence 128 --batch-size 8192 --epochs 10 --frames-per-proc 512 --visibility 15 --model FourRoomVisibility15

```


## Create dataset for imitation learning
```
python il_dataset.py --num_eps NUM_EPISODES --dir DATA_DIR [--render]
```
