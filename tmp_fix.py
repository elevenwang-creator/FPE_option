import os, re
files = ['src/engines/fpe/gpu/domain.mojo', 'src/engines/fpe/gpu/calibration.mojo', 'src/engines/fpe/gpu/matrix.mojo', 'src/engines/fpe/gpu/solver.mojo']
for f in files:
    with open(f, 'r') as fp:
        ct = fp.read()
    ct = re.sub(r'rebind\[[a-zA-Z0-9_]+\.element_type\]\((.*?)\)', r'\1', ct)
    ct = re.sub(r'rebind\[Scalar\[GPU_DTYPE\]\]\((.*?)\)', r'\1', ct)
    with open(f, 'w') as fp:
        fp.write(ct)
