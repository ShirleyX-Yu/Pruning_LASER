import subprocess
import json
import argparse
import time

def get_args_parser():
    parser = argparse.ArgumentParser(description='tuning hyperparams for the modified LASER model')
    parser.add_argument('lname',
						help='location of modification',
						type=str)
    parser.add_argument('rate',
						help='ρ = 1 - 0.1 * rate',
						type=float)
    parser.add_argument('lnum',
						help='layer number',
						type=int)
    parser.add_argument('--sweep', type=str, default='sweep.json',
                   help='sweep file')
    return parser

#  "#SBATCH --time=167:00:00\n"
def write_run(jobname, extra=''):
    with open('temp.sh', 'w') as f:
        f.write("#!/bin/bash\n"
                "#SBATCH --job-name=job_name{0}\n"
                "#SBATCH --gres=gpu:1\n"
                "#SBATCH --nodes=1\n"
                "#SBATCH --ntasks=1\n"
                "#SBATCH --cpus-per-task=1\n"
                "#SBATCH --mem-per-cpu=50G\n"
                "#SBATCH --time=2:00:00\n"
                "#SBATCH --mail-type=begin\n"
                "#SBATCH --mail-type=end\n"
                "#SBATCH --mail-user=sy9504@cs.princeton.edu\n".format(jobname))

        cmd = "python -u intervention_gptj_fever.py "
        cat = " >jobname" + ".out"
        f.write(cmd+extra+cat+'\n')

    subprocess.call('chmod +x temp.sh', shell=True)
    time.sleep(0.1)
    subprocess.call('sbatch temp.sh', shell=True)

parser = get_args_parser()
args = parser.parse_args()
print(vars(args))

with open(args.sweep, 'r') as f:
    sweep_args = json.load(f)

arglist = ['']
for opt, vals in sweep_args.items():
    new_arglist = []
    for j, v in enumerate(vals):
        for i in range(len(arglist)):
            new_arglist.append( arglist[i] + ' --'+opt+' '+str(v) )
    arglist = new_arglist

for i, ar in enumerate(arglist):
    print(i, ar)
    write_run(str(i), ar)
    print('-'*50)