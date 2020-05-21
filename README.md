# NTU ADL HW3

## Installation
Type the following command to install OpenAI Gym Atari environment.

```sh
pip install opencv-python gym gym[atari]
sh download.sh
it will download checkpoints, put into ckpt
```

Please refer to [OpenAI's page](https://github.com/openai/gym) if you have any problem while installing.

## How to run :
testing policy gradient:
* `$ python3 test.py --test_pg`

testing DQN:
* `$ python3 test.py --test_dqn`

If you want to see agent playing the game,
* `$ python3 test.py --test_[pg|dqn] --do_render`

training policy gradient:
```sh
python3 main.py --train_pg
you can run different pg mode
python3 main.py --train_pg --pg_mode ppo
```

training DQN:
```sh
python3 main.py --train_dqn
you can run different dqn mode
python3 main.py --train_dqn --dqn_mode duel
```

plot results
* `python3 plot.py`

## Code structure

```
.
├── agent_dir (all agents are placed here)
│   ├── agent.py (defined 4 required functions of the agent. DO NOT MODIFY IT)
│   ├── agent_dqn.py (DQN agent sample code)
│   └── agent_pg.py (PG agent sample code)
├── argument.py (you can add your arguments in here. we will use the default value when running test.py)
├── atari_wrapper.py (wrap the atari environment. DO NOT MODIFY IT)
├── environment.py (define the game environment in HW3, DO NOT MODIFY IT)
├── main.py (main function)
├── test.py (test script. we will use this script to test your agents. DO NOT MODIFY IT)
├── download.sh (download model weight)
├── plot.py (plot Report results)
└── Report (just a report)
```
