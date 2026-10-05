import argparse
from ..workflow import train_selected

def main(argv=None):
    p=argparse.ArgumentParser(description='Train selected cVAE from prepared fold artifacts')
    p.add_argument('--root',required=True)
    p.add_argument('--fold',type=int,required=True)
    p.add_argument('--seed',type=int,required=True)
    p.add_argument('--device',default='cpu')
    p.add_argument('--ablation',choices=['no_hr_loss','no_condition'])
    a=p.parse_args(argv)
    run,_=train_selected(a.root,a.fold,a.seed,a.device,a.ablation)
    print(run)
if __name__=='__main__': main()
