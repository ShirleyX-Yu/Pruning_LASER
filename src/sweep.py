import subprocess
import json
import argparse

def write_run(jobname, extra=''):
    with open('temp.sh', 'w') as f:
        f.write("#!/bin/bash\n"
                "#SBATCH --job-name=job_name{0}\n"
                "#SBATCH --gres=gpu:1\n"
                "#SBATCH --nodes=1\n"
                "#SBATCH --ntasks=1\n"
                "#SBATCH --cpus-per-task=1\n"
                "#SBATCH --mem-per-cpu=50G\n"
                "#SBATCH --time=167:00:00\n"
                "#SBATCH --mail-type=begin\n"
                "#SBATCH --mail-type=end\n"
                "#SBATCH --mail-user=your_email\n".format(jobname))

        cmd = "python -u train.py "
        cat = " >jobname + ".out"
        f.write(cmd+extra+cat+'\n')

    subprocess.call('chmod +x temp.sh', shell=True)
    time.sleep(0.1)
    subprocess.call('sbatch -A vertaix temp.sh', shell=True)

parser = argparse.ArgumentParser(description='')
parser.add_argument('--sweep', type=str, default='sweep.json',
                   help='sweep file')
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