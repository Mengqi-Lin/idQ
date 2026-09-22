#!/usr/bin/env python3
"""Create a separate, prespecified K=10 known-SAT ensemble."""
import json
import hashlib
from pathlib import Path
import numpy as np
from run_benchmark import preprocess
from idQ.utils import boolean_product, is_valid_factorization_counterexample

def main():
    rng=np.random.default_rng(20260923)
    H=np.zeros((10,10),dtype=int)
    for j in range(10):
        H[j,j]=H[j,(j+1)%10]=1
    cases=[]
    for J in (20,35,50):
        for rep in range(6):
            X=np.zeros((J,10),dtype=int)
            for row in X:
                row[rng.choice(10,2,replace=False)]=1
            Q=boolean_product(X,H)
            assert is_valid_factorization_counterexample(Q,X,H)
            B,branch,elapsed=preprocess(Q)
            cases.append(dict(case_id=f'p{len(cases):04d}',family='planted_sat',cell=f'J{J}_cycle_factor_w2',
                              replicate=rep,J=J,K=10,Jb=len(B),branch=branch,
                              preprocessing_seconds=elapsed,
                              sha256=hashlib.sha256(np.asarray(Q,dtype=np.uint8).tobytes()).hexdigest(),
                              Q=Q.tolist(),basis=B.tolist(),known_witness=dict(X=X.tolist(),H=H.tolist())))
    path=Path(__file__).parent/'planted_cases.json'
    path.write_text(json.dumps(dict(seed=20260923,cases=cases),indent=2))
    print(json.dumps(dict(inputs=len(cases),reaching_sat=sum(c['branch']=='sat' for c in cases),
                         branches={b:sum(c['branch']==b for c in cases) for b in {c['branch'] for c in cases}})))

if __name__=='__main__':main()
