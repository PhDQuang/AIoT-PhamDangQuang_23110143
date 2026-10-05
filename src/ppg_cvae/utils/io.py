from pathlib import Path
import json, yaml, platform, sys, os, torch

def save_json(obj,path): Path(path).parent.mkdir(parents=True,exist_ok=True); Path(path).write_text(json.dumps(obj,indent=2,default=str),encoding='utf-8')
def load_yaml(path): return yaml.safe_load(Path(path).read_text(encoding='utf-8'))
def environment_info():
    import numpy, scipy, pandas
    info={"python":sys.version,"platform":platform.platform(),"torch":str(torch.__version__),
          "numpy":numpy.__version__,"scipy":scipy.__version__,"pandas":pandas.__version__,
          "cuda":torch.cuda.is_available(),"cpu":platform.processor() or os.environ.get('PROCESSOR_IDENTIFIER','unknown'),
          "cpu_count":os.cpu_count(),"torch_threads":torch.get_num_threads()}
    try:
        if sys.platform == 'win32':
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r'HARDWARE\DESCRIPTION\System\CentralProcessor\0') as key:
                info['cpu'] = winreg.QueryValueEx(key, 'ProcessorNameString')[0].strip()
        elif Path('/proc/cpuinfo').exists():
            for line in Path('/proc/cpuinfo').read_text().splitlines():
                if line.startswith('model name'):
                    info['cpu'] = line.split(':', 1)[1].strip()
                    break
    except (OSError, ImportError):
        pass
    if torch.cuda.is_available(): info['gpu']=torch.cuda.get_device_name()
    try:
        if sys.platform=='win32':
            import ctypes
            class MemoryStatus(ctypes.Structure):
                _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(name,ctypes.c_ulonglong) for name in ('total_phys','avail_phys','total_page','avail_page','total_virtual','avail_virtual','avail_extended')]
            status=MemoryStatus(); status.length=ctypes.sizeof(status)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)): info['ram_bytes']=int(status.total_phys)
        else:
            info['ram_bytes']=int(os.sysconf('SC_PAGE_SIZE')*os.sysconf('SC_PHYS_PAGES'))
    except (AttributeError,OSError,ValueError): info['ram_bytes']=None
    return info
