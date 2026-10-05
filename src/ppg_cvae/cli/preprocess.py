import argparse
from ..data.prepare import prepare_fold

def main(argv=None):
    p = argparse.ArgumentParser(description='Prepare leakage-safe fold arrays and manifest')
    p.add_argument('--data-root',required=True)
    p.add_argument('--output-dir',required=True)
    p.add_argument('--fold',type=int,default=1)
    p.add_argument('--label-start-sample',type=int,required=True)
    p.add_argument('--alignment-source',required=True)
    a=p.parse_args(argv)
    manifest=prepare_fold(a.data_root,a.output_dir,a.fold,label_start_sample=a.label_start_sample,alignment_source=a.alignment_source)
    print(manifest.groupby('split').valid.agg(['size','sum']))
if __name__=='__main__': main()
