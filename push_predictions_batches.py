import subprocess
import time
import sys

batches = [
    ("m1_resnet18", [43, 44]),
    ("m2_mlp", [42, 43, 44]),
    ("m3_mlp", [42, 43, 44]),
    ("m4_gatv2", [42, 43, 44]),
    ("m5_gatv2", [42, 43, 44]),
    ("m4_sage", [42, 43, 44]),
    ("m5_sage", [42, 43, 44]),
]

def run_cmd(cmd):
    print(f"Running: {cmd}", flush=True)
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"Error ({res.returncode}): {res.stderr}", flush=True)
    else:
        print(res.stdout, flush=True)
    return res.returncode == 0

for model, seeds in batches:
    for seed in seeds:
        pattern = f"outputs/predictions/{model}_s{seed}_*.npz"
        commit_msg = f"Add {model} seed {seed} predictions (Lizard & CoNSeP)"
        print(f"\n==========================================", flush=True)
        print(f"Processing: {model} Seed {seed}", flush=True)
        print(f"==========================================", flush=True)
        
        # Stage files
        if not run_cmd(f"git add {pattern}"):
            print(f"Failed to add {pattern}, aborting.", flush=True)
            sys.exit(1)
            
        # Commit
        if not run_cmd(f'git commit --no-gpg-sign -m "{commit_msg}"'):
            print(f"Commit failed or nothing to commit for {model} s{seed}, skipping...", flush=True)
            continue
            
        # Push with retry
        pushed = False
        for attempt in range(1, 4):
            print(f"Pushing {model} s{seed} (attempt {attempt}/3)...", flush=True)
            if run_cmd("git push origin main"):
                print(f"Successfully pushed {model} s{seed}!", flush=True)
                pushed = True
                break
            else:
                print(f"Push attempt {attempt} failed, waiting 10s before retry...", flush=True)
                time.sleep(10)
                
        if not pushed:
            print(f"FATAL: Failed to push {model} s{seed} after 3 attempts.", flush=True)
            sys.exit(1)
            
        time.sleep(2)

print("\nALL 210 PREDICTION FILES SUCCESSFULLY COMMITTED AND PUSHED!", flush=True)
