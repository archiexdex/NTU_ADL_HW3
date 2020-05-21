def add_arguments(parser):
    '''
    Add your arguments here if needed. The TAs will run test.py to load
    your default arguments.

    For example:
        parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
        parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate for training')
    '''
    parser.add_argument('--display_freq',       type=int, default=10, help='display frequency for training')
    parser.add_argument('--gamma',              type=float, default=0.99, help='constant parameter')
    parser.add_argument('--batch_size',         type=int, default=512, help='batch size')
    
    
    
    parser.add_argument('--pg_episodes', type=int, default=10000, help='pg episode')
    parser.add_argument('--pg_lr', type=float, default=3e-3, help='pg lr')
    parser.add_argument('--pg_mode', type=str, default='pg', help='ppo or pg')
    parser.add_argument('--ppo_clip', type=float, default=0.2, help='ppo clip')
    parser.add_argument('--ppo_steps', type=int, default=5, help='ppo step')
    
    parser.add_argument('--dqn_mode',           type=str, default='dqn', help='dqn, duel, prioritized')    
    parser.add_argument('--target_update_freq', type=int, default=1000, help='target_update_freq')
    
    
    parser.add_argument('--dqn_model_path',     type=str, default='ckpt/dqn_best', help='load path for dqn')
    parser.add_argument('--duel_model_path',     type=str, default='ckpt/duel_best', help='load path for duel')
    parser.add_argument('--ppo_model_path',     type=str, default='ckpt/ppo_best', help='load path for ppo')
    parser.add_argument('--pg_model_path',     type=str, default='ckpt/pg_best', help='load path for pg')
    

    parser.add_argument('--rw_pg_path',         type=str, default='rw/rw_pg.npy', help='save path for pg reward log')
    parser.add_argument('--rw_ppo_path',         type=str, default='rw/rw_ppo.npy', help='save path for ppo reward log')
    parser.add_argument('--rw_dqn_path',        type=str, default='rw/rw_dqn.npy', help='save path for dqn reward log')
    parser.add_argument('--rw_duel_path',        type=str, default='rw/rw_duel.npy', help='save path for duel reward log')
    
    return parser
