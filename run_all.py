import subprocess

subprocess.run(['python', 'run_train.py'], check=True)
subprocess.run(['python', 'run_infer.py', '--model_path=x', '--model_mode=original',  'tile', '--input_dir=x',  '--output_dir=x'], check=True)
subprocess.run(['python', 'compute_stats.py'], check=True)