import argparse
from ..data.prepare import load_prepared
from ..training.train_hr_proxy import train_proxy
from ..evaluation.calibration import calibrate_detector

def main(argv=None):
    p=argparse.ArgumentParser(description='Train HR proxy on prepared train arrays; calibrate detector on validation')
    p.add_argument('--data',required=True,help='Prepared fold directory')
    p.add_argument('--output-dir',required=True)
    p.add_argument('--device',default='cpu')
    p.add_argument('--epochs',type=int,default=50)
    a=p.parse_args(argv)
    tx,tc=load_prepared(a.data,'train'); vx,vc=load_prepared(a.data,'val')
    _,history=train_proxy(tx,tc,vx,vc,epochs=a.epochs,device=a.device,output_dir=a.output_dir)
    print(history[-1]); print(calibrate_detector(vx,vc,a.output_dir))
if __name__=='__main__': main()
