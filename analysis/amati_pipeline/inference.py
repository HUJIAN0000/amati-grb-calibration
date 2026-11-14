from __future__ import annotations
import importlib.metadata
import json
import platform
import sys
import time
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
import emcee
from .data import ROOT,json_write,sha256


def optimize(model,start=None):
    lo,hi=model.bounds.T
    x=np.clip(np.array(model.initial if start is None else start,float),lo+1e-6,hi-1e-6)
    objective=lambda t:-model(t) if np.isfinite(model(t)) else 1e90
    best=minimize(objective,x,method='Nelder-Mead',options={'maxiter':24000,'xatol':1e-8,'fatol':1e-9})
    candidate=minimize(objective,best.x,method='L-BFGS-B',bounds=list(zip(lo+1e-8,hi-1e-8)),options={'maxiter':2000})
    if candidate.success and candidate.fun<best.fun-1e-8:best=candidate
    if not np.isfinite(best.fun) or best.fun>1e50:raise RuntimeError('Optimization failed to find a finite posterior')
    return best


def summaries(chain,names):
    q=np.percentile(chain,[2.5,16,50,84,97.5],axis=0)
    return {name:{'median':float(q[2,i]),'minus':float(q[2,i]-q[1,i]),'plus':float(q[3,i]-q[2,i]),
                  'ci68':[float(q[1,i]),float(q[3,i])],'ci95':[float(q[0,i]),float(q[4,i])]}
            for i,name in enumerate(names)}


def chain_summary(model,chain,nburn,acceptance):
    kept=chain[nburn:];flat=kept.reshape(-1,kept.shape[-1])
    tau=emcee.autocorr.integrated_time(kept,tol=0)
    ratio=len(kept)/tau
    result={'parameters':summaries(flat,model.names),'tau_postburn':dict(zip(model.names,tau.tolist())),
            'postburn_steps_div_tau':dict(zip(model.names,ratio.tolist())),
            'effective_sample_size':dict(zip(model.names,(flat.shape[0]/tau).tolist())),
            'convergence_50tau':bool(np.all(ratio>=50)),'acceptance_fraction_mean':float(np.mean(acceptance)),
            'acceptance_fraction_range':[float(np.min(acceptance)),float(np.max(acceptance))],
            'steps':len(chain),'burn':nburn,'walkers':chain.shape[1],'n_samples':len(flat),
            'prior_boundary_fraction':{name:{'lower_1pct':float(np.mean(flat[:,i]<lo+.01*(hi-lo))),
                                             'upper_1pct':float(np.mean(flat[:,i]>hi-.01*(hi-lo)))}
                                       for i,(name,(lo,hi)) in enumerate(zip(model.names,model.bounds))}}
    return result,flat


def run(model,out,settings,resume=False):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    cfg={'model':model.config,'sampling':settings,'parameter_names':model.names,'bounds':model.bounds.tolist(),
         'input_manifest_sha256':sha256(model.data.source/'manifest.json') if (model.data.source/'manifest.json').exists() else None}
    if model.config.get('anchor_hosts'):cfg['anchor_hosts_sha256']=sha256(model.data.source / model.config['anchor_hosts'])
    configfile=out/'config.json';backendfile=out/'chain.h5'
    if backendfile.exists() and not resume:raise FileExistsError(f'{backendfile} exists. Use --resume or choose a new output.')
    if resume and configfile.exists() and json.loads(configfile.read_text(encoding='utf-8'))!=cfg:
        raise ValueError('Resume configuration differs from frozen configuration')
    json_write(configfile,cfg)
    result=optimize(model)
    best=np.asarray(result.x)
    json_write(out/'bestfit.json',{'parameters':dict(zip(model.names,best.tolist())),'lnL':float(model(best)),
                                 'lnL_parts':model.log_likelihood(best,True)[0].tolist(),'optimizer_success':bool(result.success)})
    print(f'MAP {out.name}: lnL={model(best):.6f} '+str(dict(zip(model.names,np.round(best,5)))),flush=True)
    steps=int(settings.get('steps',7000));burn=int(settings.get('burn',1800))
    walkers=int(settings.get('walkers',96));seed=int(settings.get('seed',20261023))
    maxsteps=int(settings.get('max_steps',steps));chunk=int(settings.get('chunk',1000))
    if walkers<2*len(best) or burn>=steps:raise ValueError('Invalid walkers or burn')
    backend=emcee.backends.HDFBackend(str(backendfile))
    sampler=emcee.EnsembleSampler(walkers,len(best),model,vectorize=True,backend=backend)
    saved_summary=out/'summary.json'
    if resume and backend.iteration>=steps and saved_summary.exists():
        saved=json.loads(saved_summary.read_text(encoding='utf-8'))
        if saved.get('steps')==backend.iteration and (not settings.get('auto_extend',False)
                or (saved.get('convergence_50tau') and saved.get('tau_stable_5pct'))):
            print(f'{out.name}: existing completed chain verified; no samples overwritten.',flush=True)
            return saved
    if resume and backend.iteration:
        state=None
    else:
        backend.reset(walkers,len(best))
        rng=np.random.default_rng(seed)
        scales=np.array([.03 if 'sigma' in n else .015 if n in {'b1','a1'} else .04 for n in model.names])
        state=np.empty((walkers,len(best)))
        for i in range(walkers):
            for attempt in range(10000):
                trial=best+scales*rng.normal(size=len(best))
                if np.isfinite(model(trial)):state[i]=trial;break
            else:raise RuntimeError('Walker initialization failed')
        sampler.random_state=np.random.RandomState(seed).get_state()
    started=time.perf_counter();previous_tau=None;status=None
    target=steps
    if sampler.iteration>=target and settings.get('auto_extend',False) and sampler.iteration<maxsteps:
        previous_tau=emcee.autocorr.integrated_time(sampler.get_chain(discard=burn),tol=0)
        target=min(maxsteps,sampler.iteration+2000)
    while sampler.iteration<target:
        count=min(chunk,target-sampler.iteration)
        state=sampler.run_mcmc(state,count,progress=False)
        print(f'{out.name}: {sampler.iteration}/{target} steps, {time.perf_counter()-started:.1f}s',flush=True)
        json_write(out/'progress.json',{'steps':sampler.iteration,'target':target,'seconds_this_invocation':time.perf_counter()-started})
        if sampler.iteration>=target:
            status,flat=chain_summary(model,sampler.get_chain(),burn,sampler.acceptance_fraction)
            tau=np.array(list(status['tau_postburn'].values()))
            stable=previous_tau is not None and np.all(abs(tau-previous_tau)/tau<.05)
            status['tau_stability_checked']=previous_tau is not None;status['tau_stable_5pct']=bool(stable)
            if settings.get('auto_extend',False) and (not status['convergence_50tau'] or not stable) and target<maxsteps:
                previous_tau=tau
                target=min(maxsteps,max(target+2000,int(burn+55*max(tau))))
    if status is None:status,flat=chain_summary(model,sampler.get_chain(),burn,sampler.acceptance_fraction)
    status['lnL_map']=float(model(best));status['sampling_seconds_this_invocation']=time.perf_counter()-started
    status['status']='passed_50tau' if status['convergence_50tau'] else 'insufficient_chain_length'
    status['environment']={'python':sys.version,'platform':platform.platform(),
                           'packages':{p:importlib.metadata.version(p) for p in ['numpy','scipy','pandas','emcee','matplotlib','h5py']}}
    json_write(out/'summary.json',status)
    # Complete time/walker chain and log probability stay in HDF5. Compact flat samples for figures.
    stride=max(1,len(flat)//30000)
    np.savez_compressed(out/'posterior_plot_samples.npz',samples=flat[::stride],names=np.array(model.names))
    report=['group,parameter,median,minus,plus,lower95,upper95']
    for name,s in status['parameters'].items():
        report.append(f"parameters,{name},{s['median']},{s['minus']},{s['plus']},{s['ci95'][0]},{s['ci95'][1]}")
    (out/'posterior_summary.csv').write_text('\n'.join(report)+'\n',encoding='utf-8')
    print(f'{out.name}: {status["status"]}',flush=True)
    return status
