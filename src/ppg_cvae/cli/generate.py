"""CPU free-generation CLI."""
import argparse, json, numpy as np, torch
from pathlib import Path
from ..models.cvae import Decoder
from ..evaluation.heart_rate import estimate_hr
from ..utils.seed import seed_everything

def main(argv=None):
    ap=argparse.ArgumentParser(); ap.add_argument('--checkpoint',required=True); ap.add_argument('--hr',type=float,required=True); ap.add_argument('--seed',type=int,default=42); ap.add_argument('--num-samples',type=int,default=10); ap.add_argument('--output',required=True); ap.add_argument('--latent-dim',type=int,default=16); ap.add_argument('--condition-mode',choices=['hr','zero'],default='hr'); ap.add_argument('--detector-config'); args=ap.parse_args(argv)
    if args.hr <= 0 or args.num_samples < 1: ap.error('HR and num-samples must be positive')
    detector=json.loads(Path(args.detector_config).read_text(encoding='utf-8')) if args.detector_config else {}
    seed_everything(args.seed); dec=Decoder(args.latent_dim); dec.load_state_dict(torch.load(args.checkpoint,map_location='cpu',weights_only=True)); dec.eval(); c=torch.full((args.num_samples,),0. if args.condition_mode=='zero' else args.hr/60.,dtype=torch.float32); z=torch.randn(args.num_samples,args.latent_dim)
    with torch.inference_mode(): y=dec(z,c).numpy()
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    measured=[estimate_hr(s,**detector) for s in y[:,0]]; np.savez(args.output,ppg=y,target_hr=args.hr,measured_hr=np.array([m.hr if m.measurable else np.nan for m in measured]),measurable=np.array([m.measurable for m in measured]),condition_mode=args.condition_mode,detector_config=json.dumps(detector))
    print({'target_hr':args.hr,'num_samples':args.num_samples,'measured_hr':[m.hr for m in measured],'measurable':[m.measurable for m in measured],'output':args.output})
if __name__=='__main__': main()
