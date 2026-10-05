import argparse
from ..workflow import evaluate_run

def main(argv=None):
    p=argparse.ArgumentParser(description='Evaluate final artifacts with locked protocol')
    p.add_argument('--root',required=True)
    p.add_argument('--fold',type=int,required=True)
    p.add_argument('--seed',type=int,required=True)
    p.add_argument('--method',choices=['cvae','gaussian','no_hr_loss','no_condition'],default='cvae')
    p.add_argument('--device',default='cpu')
    p.add_argument('--protocol-locked',action='store_true',help='Dataset alignment and evaluation rules are finalized for all folds')
    a=p.parse_args(argv)
    if not a.protocol_locked: p.error('Finalize the protocol for all folds before test evaluation; pass --protocol-locked after doing so')
    print(evaluate_run(a.root,a.fold,a.seed,a.method,a.device).to_string(index=False))
if __name__=='__main__': main()
