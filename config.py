import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Audio-Visual Synchronization Detection')
    
    # Run name (and checkpoint filename)
    parser.add_argument('--name', type=str, default='first_run',
                        help='Name of the run (also used as filename for the checkpoint)')

    # Model configuration
    parser.add_argument('--tau', type=int, default=15,
                        help='Temporal window size (left and right distance from the central frame)')
    
    # Training configuration
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=500,
                        help='Number of training epochs')
    parser.add_argument('--early_stopping_patience', type=int, default=10,
                        help='Stop after this many epochs without improvement')
    parser.add_argument('--scheduler_patience', type=int, default=5,
                        help='Patience for learning rate scheduler')
    parser.add_argument('--learning_rate', type=float, default=1e-5,
                        help='Initial learning rate')
    parser.add_argument('--penalty_coefficient', type=float, default=0.1,
                        help='Coefficient for the output magnitude regularizer loss term')
    parser.add_argument('--discrete_datapoints', action='store_false', help="each datapoint has features in a seperate file")
    
    # Logging configuration
    parser.add_argument('--use_tqdm', action='store_true',
                        help='Use tqdm progress bars')
    parser.add_argument('--no_intermediate_logging', action='store_true',
                        help='Disable intermediate logging')
    parser.add_argument('--log_interval', type=int, default=1000,
                        help='Interval for intermediate logging')
    parser.add_argument('--save_path', type=str, default='checkpoints/',
                        help='Path to save model checkpoints')
    
    # Data paths
    parser.add_argument('--data_root_path', type=str, default="av1m_features/",
                        help='Root directory for feature data')
    parser.add_argument('--metadata_root_path',  type=str, default="av1m_metadata/",
                        help='Metadata path directory for feature data')
    
    args = parser.parse_args()

    return args