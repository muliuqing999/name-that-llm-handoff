def run(cmd, env=None):
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env, bufsize=1)
    for line in proc.stdout:
        if not re.search(r'Warning|warn', line):
            print(line, end='')
    return proc.wait()


COMP_SEC = 0.0
if not FAST_COMMIT:
    t0 = time.time()
    code = run([sys.executable, 'comp_stack.py'], env=dict(os.environ, NTL_DROP_Q35='1', NTL_PIPE_WORKERS='1'))
    COMP_SEC = time.time() - t0
    print('comp stack exit code', code, 'wall', round(COMP_SEC), 's')
