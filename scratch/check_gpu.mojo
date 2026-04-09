from std.sys import has_accelerator
from std.sys.info import has_apple_gpu_accelerator
from std.gpu.host import DeviceContext

def main() raises:
    print("Has accelerator:", has_accelerator())
    print("Has Apple GPU accelerator:", has_apple_gpu_accelerator())
    
    var ctx = DeviceContext()
    print("Successfully created DeviceContext")
