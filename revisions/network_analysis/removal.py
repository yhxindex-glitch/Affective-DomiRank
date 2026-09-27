"""Evaluate fixed node-removal orders on the original case-study graph.

The removal settings and reference distributions are specified in
removal_protocol.json. Input scores remain fixed during node deletion.
"""
from pathlib import Path
from xml.etree import ElementTree as ET
from collections import defaultdict
import csv, hashlib, json, platform, time
import numpy as np

BASE=Path(__file__).resolve().parents[2]
WORK=Path(__file__).resolve().parent
OUT=BASE/'results/node_removal'
SOURCE=BASE/'data/processed/networks/network_2020.graphml'
CASE=BASE/'analysis/results/energy_minerals_case_study.json'
CASE_SCRIPT=BASE/'experiments/run_case_study_energy_subnetwork.py'
NS={'g':'http://graphml.graphdrawing.org/xmlns'}
FIELDS={'Affective DomiRank':'affective_domirank','Traditional DomiRank':'domirank'}
METHODS=list(FIELDS)+['Total degree','PageRank']

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,obj):Path(p).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def save_csv(path,rows):
    with Path(path).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def norm(a):
    lo=min(a);hi=max(a)
    return np.zeros(len(a)) if hi-lo<1e-12 else (np.asarray(a)-lo)/(hi-lo)
def read_graph():
    r=ET.parse(SOURCE).getroot()
    keys={e.get('id'):e.get('attr.name') for e in r.findall('g:key',NS)}
    g=r.find('g:graph',NS)
    nodes={e.get('id'):{keys[d.get('key')]:d.text for d in e.findall('g:data',NS)} for e in g.findall('g:node',NS)}
    nodes={n:a for n,a in nodes.items() if a.get('industry') in ['Energy','Minerals']}
    names=list(nodes);ix={n:i for i,n in enumerate(names)}
    B=np.zeros((len(names),len(names)),dtype=bool)
    for e in g.findall('g:edge',NS):
        a=e.get('source');b=e.get('target')
        if a in ix and b in ix:B[ix[a],ix[b]]=True
    return names,nodes,B
def masks(B):
    return [sum(1<<int(j) for j in np.flatnonzero(row)) for row in B]
def reachable(start,active,adj):
    seen=0;front=start
    while front:
        seen|=front;neighbors=0
        while front:
            bit=front&-front;front^=bit;neighbors|=adj[bit.bit_length()-1]
        front=neighbors&active&~seen
    return seen
def components(active,adj,out,inc):
    rem=active;wc=[]
    while rem:
        got=reachable(rem&-rem,active,adj);wc.append(got.bit_count());rem&=~got
    rem=active;sc=[]
    while rem:
        bit=rem&-rem
        got=reachable(bit,active,out)&reachable(bit,active,inc)
        assert got&bit
        sc.append(got.bit_count());rem&=~got
    return max(wc,default=0),max(sc,default=0)
def curve(order,B,precomputed=None):
    n=len(B);assert sorted(order)==list(range(n))
    adj,out,inc=precomputed or (masks(B|B.T),masks(B),masks(B.T))
    active=(1<<n)-1;res=[components(active,adj,out,inc)]
    for j in order:
        active &= ~(1<<int(j));res.append(components(active,adj,out,inc))
    result=np.asarray(res,dtype=np.int16)
    assert result.shape==(n+1,2) and np.all(np.diff(result,axis=0)<=0)
    assert np.all(result[-1]==0) and np.all(result[:,1]<=result[:,0])
    return result
def closure_components(B):
    if not len(B):return np.zeros(2,dtype=int)
    reach=B.copy();weak=B|B.T
    np.fill_diagonal(reach,True);np.fill_diagonal(weak,True)
    for j in range(len(B)):
        reach |= reach[:,j:j+1]&reach[j:j+1,:]
        weak |= weak[:,j:j+1]&weak[j:j+1,:]
    return np.array([weak.sum(axis=1).max(),(reach&reach.T).sum(axis=1).max()])
def verify_curve(order,B,values):
    active=list(range(len(B)))
    assert np.array_equal(closure_components(B),values[0])
    for k,j in enumerate(order,1):
        active.remove(j)
        assert np.array_equal(closure_components(B[np.ix_(active,active)]),values[k]),k
def pagerank(B):
    n=len(B);d=B.sum(axis=1);T=np.zeros((n,n))
    T[d>0]=B[d>0]/d[d>0,None];T[d==0]=1/n
    p=np.full(n,1/n)
    for count in range(10000):
        new=.85*(T.T@p)+.15/n
        if np.sum(abs(new-p))<1e-12:
            assert abs(new.sum()-1)<1e-10
            return new,count+1
        p=new
    raise RuntimeError('PageRank did not converge')
def summarize(v):
    n=len(v)-1;q=np.arange(n+1)/n
    out={}
    for j,key in enumerate(['weak','strong']):
        vals=v[:,j];g0=float(vals[0]);y=vals/g0
        out[key]={'initial_size':int(g0),'auc_initial_normalized':float(np.trapezoid(y,q)),
          'auc_N_normalized':float(np.trapezoid(vals/n,q)),
          'after10':int(vals[min(10,n)]),'after20':int(vals[min(20,n)]),
          'remove_to_half_initial':int(np.flatnonzero(y<=.5)[0]),
          'remove_to_tenth_initial':int(np.flatnonzero(y<=.1)[0])}
    return out
def permute_ties(scores,rng):
    groups=defaultdict(list)
    for i,s in enumerate(norm(scores)):groups[round(float(s),12)].append(i)
    result=[]
    for s in sorted(groups,reverse=True):result.extend(int(i) for i in rng.permutation(groups[s]))
    return result
def dist_summary(x):
    return {'mean':float(np.mean(x)),'sd':float(np.std(x,ddof=1)),
      'p025':float(np.quantile(x,.025)),'median':float(np.median(x)),
      'p975':float(np.quantile(x,.975)),'min':float(np.min(x)),'max':float(np.max(x))}

def main():
    started=time.perf_counter();OUT.mkdir(parents=True,exist_ok=True)
    protocol=json.loads((WORK/'removal_protocol.json').read_text('utf-8'))
    protected=[SOURCE,CASE,CASE_SCRIPT,WORK/'removal_protocol.json']
    hashes={str(p):sha(p) for p in protected}
    names,attrs,B=read_graph();n=len(names)
    assert n==112 and int(B.sum())==5075
    assert not np.any(np.diag(B))
    original=json.loads(CASE.read_text('utf-8-sig'))
    original_scores={m:np.array([float(attrs[k][field]) for k in names]) for m,field in FIELDS.items()}
    original_order={m:sorted(range(n),key=lambda i:float(norm(s)[i]),reverse=True) for m,s in original_scores.items()}
    tie_order_differences=[]
    for row in original['affective_top10']+original['largest_rank_promotions']:
        i=names.index(row['node'])
        for m,prefix in [('Affective DomiRank','affective'),('Traditional DomiRank','traditional')]:
            assert abs(norm(original_scores[m])[i]-row[prefix+'_score'])<1e-12
            current_rank=original_order[m].index(i)+1;scores=norm(original_scores[m])
            if current_rank!=row[prefix+'_rank']:
                rounded=np.round(scores,12);lo=int(np.sum(rounded>rounded[i]))+1;hi=int(np.sum(rounded>=rounded[i]))
                assert lo<=row[prefix+'_rank']<=hi and lo<=current_rank<=hi
                tie_order_differences.append({'method':m,'node':row['node'],'saved_rank':row[prefix+'_rank'],'current_rank':current_rank,'equal_score_rank_interval':[lo,hi]})
    assert [names[i] for i in original_order['Affective DomiRank'][:10]]==[r['node'] for r in original['affective_top10']]
    for row in original['overlap_summary']:
        k=row['k'];assert len(set(original_order['Affective DomiRank'][:k])&set(original_order['Traditional DomiRank'][:k]))==row['overlap_count']
    results={'protocol':protocol,'protocol_sha256':hashes[str(WORK/'removal_protocol.json')],
      'source_hashes':hashes,'runtime':{'python':platform.python_version(),'numpy':np.__version__},
      'case_reproduction_verified':True,'within_tie_order_differences':tie_order_differences,'scopes':{}}
    curve_rows=[];summary_rows=[];ranking_rows=[];tie_rows=[]
    sectors=['Combined','Energy','Minerals']
    random_seeds=np.random.SeedSequence(protocol['random_reference']['seed']).spawn(3)
    tie_seeds=np.random.SeedSequence(protocol['tie_sensitivity']['seed']).spawn(3)
    for scope_index,scope in enumerate(sectors):
        indices=[i for i in range(n) if scope=='Combined' or attrs[names[i]]['industry']==scope]
        snames=[names[i] for i in indices];G=B[np.ix_(indices,indices)];N=len(G)
        pre=(masks(G|G.T),masks(G),masks(G.T))
        sc={'node_count':N,'directed_edge_count':int(G.sum()),'methods':{},'random_reference':{},'tie_sensitivity':{}}
        scores={m:s[indices] for m,s in original_scores.items()}
        scores['Total degree']=G.sum(axis=0)+G.sum(axis=1)
        scores['PageRank'],iterations=pagerank(G)
        sc['pagerank_iterations']=iterations
        print(f'{scope}: {N} nodes, {int(G.sum())} directed edges',flush=True)
        for m in METHODS:
            if m in FIELDS:
                global_names=[names[i] for i in original_order[m]]
                order=sorted(range(N),key=lambda i:global_names.index(snames[i]))
            else:order=sorted(range(N),key=lambda i:float(scores[m][i]),reverse=True)
            v=curve(order,G,pre);verify_curve(order,G,v)
            summary=summarize(v);sc['methods'][m]=summary
            for k,i in enumerate(order,1):ranking_rows.append({'scope':scope,'method':m,'rank':k,'node':snames[i],'score':float(scores[m][i])})
            for k,row in enumerate(v):
                curve_rows.append({'scope':scope,'method':m,'removed':k,'removed_fraction':k/N,
                  'largest_weak_count':int(row[0]),'largest_strong_count':int(row[1]),
                  'weak_initial_fraction':float(row[0]/v[0,0]),'strong_initial_fraction':float(row[1]/v[0,1]),
                  'weak_original_N_fraction':float(row[0]/N),'strong_original_N_fraction':float(row[1]/N)})
            for metric,sm in summary.items():summary_rows.append({'scope':scope,'method':m,'metric':metric,**sm})
        # Random complete removal orders: reference distribution on this fixed graph.
        rng=np.random.default_rng(random_seeds[scope_index])
        random_values=np.stack([curve([int(i) for i in rng.permutation(N)],G,pre) for _ in range(protocol['random_reference']['permutations'])])
        np.savez_compressed(OUT/f'{scope.lower()}_random_curves.npz',counts=random_values)
        q=np.arange(N+1)/N
        random_auc=np.trapezoid(random_values/random_values[:,0:1,:],q,axis=1)
        for j,key in enumerate(['weak','strong']):
            sc['random_reference'][key]={'auc_initial_normalized':dist_summary(random_auc[:,j]),
              'after10':dist_summary(random_values[:,min(10,N),j]),'after20':dist_summary(random_values[:,min(20,N),j])}
        for statistic,values in [('mean',random_values.mean(axis=0)),('p025',np.quantile(random_values,.025,axis=0)),('p975',np.quantile(random_values,.975,axis=0))]:
            for k,row in enumerate(values):curve_rows.append({'scope':scope,'method':'Random '+statistic,'removed':k,'removed_fraction':k/N,
              'largest_weak_count':float(row[0]),'largest_strong_count':float(row[1]),
              'weak_initial_fraction':float(row[0]/values[0,0]),'strong_initial_fraction':float(row[1]/values[0,1]),
              'weak_original_N_fraction':float(row[0]/N),'strong_original_N_fraction':float(row[1]/N)})
        # Numerical tie sensitivity is kept separate from the exact saved ranking.
        rng=np.random.default_rng(tie_seeds[scope_index])
        for m in METHODS:
            count=protocol['tie_sensitivity']['permutations']
            tv=np.stack([curve(permute_ties(scores[m],rng),G,pre) for _ in range(count)])
            ta=np.trapezoid(tv/tv[:,0:1,:],q,axis=1)
            tie_groups=defaultdict(list)
            for i,s in enumerate(norm(scores[m])):tie_groups[round(float(s),12)].append(snames[i])
            sc['tie_sensitivity'][m]={'tie_groups':[v for v in tie_groups.values() if len(v)>1],
              'weak_auc':dist_summary(ta[:,0]),'strong_auc':dist_summary(ta[:,1])}
            for j,key in enumerate(['weak','strong']):tie_rows.append({'scope':scope,'method':m,'metric':key,**dist_summary(ta[:,j])})
        results['scopes'][scope]=sc
        print(json.dumps({'scope':scope,'main_results':{m:sc['methods'][m] for m in FIELDS},'random':sc['random_reference']},ensure_ascii=False),flush=True)
    assert all(sha(p)==h for p,h in hashes.items())
    results['verification']={'input_files_unchanged':True,'all_deterministic_curves_independently_verified':True,
      'monotonicity_and_endpoints_verified':True,'training_runs':0,'elapsed_seconds':time.perf_counter()-started}
    dump(OUT/'results.json',results)
    save_csv(OUT/'removal_curves.csv',curve_rows);save_csv(OUT/'summary_metrics.csv',summary_rows)
    save_csv(OUT/'removal_orders.csv',ranking_rows);save_csv(OUT/'tie_sensitivity.csv',tie_rows)
    print('ANALYSIS_COMPLETE',json.dumps(results['verification']),flush=True)

if __name__=='__main__':main()
