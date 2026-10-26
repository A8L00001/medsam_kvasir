"""Run the selected models at 0/5/10/20 pct box perturbation; defaults to E0 E1.

Each experiment is delegated to 03_evaluate.py, then 05_report.py redraws the curves.
"""
import argparse
import subprocess
import sys
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--experiments',nargs='+',choices=['E0','E1','E2','E3','E3C','E4'],default=['E0','E1'])
    p.add_argument('--split',choices=['val','test'],default='val')
    p.add_argument('--final-test',action='store_true')
    p.add_argument('--trials',type=int,default=3)
    a=p.parse_args()
    if a.split=='test' and not a.final_test:raise RuntimeError('必须明确 --final-test 才能使用测试集')
    script=Path(__file__).with_name('03_evaluate.py')
    for exp in a.experiments:
        for level in (0,5,10,20):
            cmd=[sys.executable,str(script),'--exp',exp,'--split',a.split,
                 '--noise',str(level/100),'--trials',str(a.trials)]
            if a.final_test:cmd.append('--final-test')
            print('RUN:', ' '.join(cmd),flush=True);subprocess.run(cmd,check=True)
    subprocess.run([sys.executable,str(Path(__file__).with_name('05_report.py')),
                    '--split',a.split],check=True)

if __name__=='__main__':main()
