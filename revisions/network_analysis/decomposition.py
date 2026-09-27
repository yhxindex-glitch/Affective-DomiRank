"""Read-only decomposition of the existing 2020 full-network case study.

No model training, source writes, changed parameters or reranking input data.
"""
from pathlib import Path
from collections import defaultdict
from xml.etree import ElementTree as ET
import hashlib
import json
import csv
import numpy as np

BASE=Path(__file__).resolve().parents[2]
WORK=BASE/'results/network_analysis'
WORK.mkdir(parents=True,exist_ok=True)
PROJECT=BASE
SOURCE=PROJECT/'data/processed/networks/network_2020.graphml'
CASE=PROJECT/'analysis/results/case_study_attribute_baseline.json'
STABILITY=WORK/'stability_check.json'
ALPHA,BETA,GAMMA,THETA=0.01,0.5,0.1,1.0
NS={'g':'http://graphml.graphdrawing.org/xmlns'}

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def dump(path,obj):path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def rank_interval(scores,value,decimals=12):
    values=np.round(np.asarray(scores),decimals);v=round(float(value),decimals)
    lo=int(np.sum(values>v))+1;hi=int(np.sum(values>=v))
    return str(lo) if lo==hi else f'{lo}–{hi}'

def main():
    root=ET.parse(SOURCE).getroot()
    keys={e.attrib['id']:e.attrib['attr.name'] for e in root.findall('g:key',NS)}
    graph=root.find('g:graph',NS)
    nodes={e.attrib['id']:{keys[d.attrib['key']]:d.text for d in e.findall('g:data',NS)} for e in graph.findall('g:node',NS)}
    node_order=list(nodes)
    assert len(nodes)==1200
    groups=defaultdict(list)
    for name in nodes:groups[name.rsplit('_',1)[1]].append(name)
    assert len(groups)==20 and all(len(v)==60 for v in groups.values())
    matrices={g:np.zeros((len(names),len(names))) for g,names in groups.items()}
    ix={n:(g,i) for g,names in groups.items() for i,n in enumerate(names)}
    for edge in graph.findall('g:edge',NS):
        g,i=ix[edge.attrib['source']];h,j=ix[edge.attrib['target']]
        assert g==h;matrices[g][i,j]=1.0
    records={};eigs=[];residuals=[];direct_errors=[];stored_errors=[]
    for group,names in groups.items():
        B=matrices[group];M=BETA*np.eye(len(names))+ALPHA*B.T
        polarity=np.array([float(nodes[n]['sentiment_mean']) for n in names])
        s=np.where(np.abs(polarity)<.05,0,np.where(polarity<0,np.abs(polarity)**.8,-.5*np.abs(polarity)**.8))
        mask=np.array([nodes[n].get('fill_strategy','') in ('temporal_interp','temporal_decay','temporal_forward') for n in names])
        s=np.where(mask,0,s)
        base_rhs=ALPHA*THETA*B.T@np.ones(len(names))
        sentiment_rhs=GAMMA*s
        structure=np.linalg.solve(M,base_rhs)
        sentiment=np.linalg.solve(M,sentiment_rhs)
        total=structure+sentiment
        stored_base=np.array([float(nodes[n]['domirank']) for n in names])
        stored_aff=np.array([float(nodes[n]['affective_domirank']) for n in names])
        local_sentiment=GAMMA/BETA*s
        local_structure=ALPHA/BETA*B.T@(THETA-np.array(stored_aff))
        residuals.extend(M@total-base_rhs-sentiment_rhs)
        stored_errors.extend(total-stored_aff)
        direct_errors.extend(local_structure+local_sentiment-stored_aff)
        assert np.max(np.abs(stored_base-structure))<1e-12
        eigs.extend(np.linalg.eigvals(-M))
        for i,name in enumerate(names):
            records[name]={'node':name,'structure':float(structure[i]),'sentiment_net':float(sentiment[i]),
                'total':float(total[i]),'stored_total':float(stored_aff[i]),'stored_traditional':float(stored_base[i]),
                'sentiment_net_share_pct':float(100*sentiment[i]/total[i]) if abs(total[i])>1e-12 else None,
                'local_structure_eq16':float(local_structure[i]),'local_sentiment_eq16':float(local_sentiment[i]),
                'sentiment_mapping_at_centrality_stage':float(s[i]),'later_temporal_fill':bool(mask[i])}
    assert max(abs(x) for x in residuals)<1e-12
    assert max(abs(x) for x in stored_errors)<1e-12
    assert max(abs(x) for x in direct_errors)<1e-12
    # Use the existing normalized-score ordering from the case script; do not
    # rank re-solved scores, whose machine-precision tie order may differ.
    original=json.loads(CASE.read_text('utf-8-sig'))
    ranking={};normalized={}
    for field in ['domirank','affective_domirank']:
        values={n:float(nodes[n][field]) for n in node_order}
        lo=min(values.values());hi=max(values.values())
        normalized[field]={n:(v-lo)/(hi-lo) for n,v in values.items()}
        ranking[field]=sorted(node_order,key=lambda n:normalized[field][n],reverse=True)
    original_top10=original['affective_top10']
    for row in original_top10:
        n=row['node']
        assert abs(normalized['affective_domirank'][n]-row['affective_score'])<1e-12
        assert abs(normalized['domirank'][n]-row['traditional_score'])<1e-12
    k_check=[]
    for row in original['overlap_summary']:
        k=row['k'];overlap=set(ranking['domirank'][:k])&set(ranking['affective_domirank'][:k])
        assert len(overlap)==row['traditional_affective_overlap'],(k,len(overlap),row)
        k_check.append({'k':k,'overlap':len(overlap)})
    top_names=ranking['affective_domirank'][:20]
    base_values=[float(nodes[n]['domirank']) for n in node_order]
    aff_values=[float(nodes[n]['affective_domirank']) for n in node_order]
    top=[]
    for i,n in enumerate(top_names):
        row=dict(records[n]);row.update({'affective_order':i+1,
            'affective_rank_interval':rank_interval(aff_values,row['stored_total']),
            'traditional_rank_interval':rank_interval(base_values,row['stored_traditional'])})
        top.append(row)
    shares=np.array([r['sentiment_net_share_pct'] for r in top])
    base_top20=np.array([float(nodes[n]['domirank']) for n in ranking['domirank'][:20]])
    # Rank correlation to the local sentiment-only score; average ranks handle ties.
    def average_ranks(a):
        vals=np.round(a,12);unique=np.unique(vals)
        result=np.empty(len(vals),float)
        for v in unique:
            mask=vals==v;result[mask]=np.sum(vals<v)+(np.sum(mask)+1)/2
        return result
    mapped=np.array([records[n]['sentiment_mapping_at_centrality_stage'] for n in node_order])
    rho=float(np.corrcoef(average_ranks(mapped),average_ranks(np.array(aff_values)))[0,1])
    topcut=round(float(nodes[top_names[-1]]['affective_domirank']),12)
    cutoff_ties=[n for n in node_order if round(float(nodes[n]['affective_domirank']),12)==topcut]
    chosen_ties=[n for n in top_names if n in cutoff_ties]
    stability=json.loads(STABILITY.read_text('utf-8'))
    # Cached five-year checks are reused only while their evidence hashes match.
    assert sha(SOURCE)==next(r['sha256'] for r in stability['years'] if r['year']==2020)
    assert all(sha(Path(p))==h for p,h in stability['source_code_hashes'].items())
    assert all(sha(Path(r['source']))==r['sha256'] for r in stability['years'])
    bounds=[BETA/(-r['adjacency_min_real_eigenvalue']) for r in stability['years']]
    result={'scope':'2020 full-network Affective DomiRank Top-20 from the existing case study',
        'parameters':{'alpha':ALPHA,'beta':BETA,'gamma':GAMMA,'theta':THETA},
        'decomposition':'M=beta I+alpha B.T; structural=M^-1(alpha theta B.T 1); net sentiment=M^-1(gamma s); total=sum',
        'local_eq16_decomposition':'structure_local=(alpha/beta) B.T (theta 1-Gamma_aff); sentiment_local=(gamma/beta)s',
        'forcing_stage':'Use fixed sentiment forcing at original centrality calculation, preceding temporal filling, as already verified for R1.5',
        'source_hashes':{str(p):sha(p) for p in [SOURCE,CASE,STABILITY]},
        'nodes':len(nodes),'verified_overlap_counts':k_check,
        'top20_common_nodes':sorted(set(top_names)&set(ranking['domirank'][:20])),
        'top20_rows':top,
        'summary':{'structure_range':[min(r['structure'] for r in top),max(r['structure'] for r in top)],
            'net_sentiment_range':[min(r['sentiment_net'] for r in top),max(r['sentiment_net'] for r in top)],
            'total_range':[min(r['total'] for r in top),max(r['total'] for r in top)],
            'net_sentiment_share_pct_range':[float(min(shares)),float(max(shares))],
            'net_sentiment_share_pct_mean':float(np.mean(shares)),
            'net_sentiment_share_pct_median':float(np.median(shares)),
            'structure_larger_than_net_sentiment_count':sum(r['structure']>abs(r['sentiment_net']) for r in top),
            'traditional_top20_score_range':[float(min(base_top20)),float(max(base_top20))],
            'traditional_top20_score_span':float(np.ptp(base_top20)),
            'local_sentiment_only_spearman_with_affective':rho,
            'distinct_local_sentiment_scores_in_top20':sorted(set(r['sentiment_mapping_at_centrality_stage'] for r in top)),
            'top20_cutoff_tie_group':cutoff_ties,'selected_cutoff_ties':chosen_ties,
            'top20_boundary_splits_tie':len(chosen_ties)!=len(cutoff_ties)},
        'verification':{'max_reconstruction_error_vs_saved':float(max(abs(x) for x in stored_errors)),
            'max_linear_system_residual':float(max(abs(x) for x in residuals)),
            'max_local_eq16_reconstruction_error':float(max(abs(x) for x in direct_errors)),
            '2020_max_real_jacobian_eigenvalue':float(max(z.real for z in eigs)),
            'five_year_stability_evidence_hashes_match':True,
            'alpha_upper_bound_by_year':dict(zip([str(r['year']) for r in stability['years']],bounds)),
            'new_training_runs':0,'source_files_modified':False},
        'ranking_note':'Existing saved-score ordering retained. Displayed rank intervals use scores rounded to 12 decimals; no new ranking data substituted.'}
    dump(WORK/'decomposition.json',result)
    with (WORK/'top20_decomposition.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(top[0]));writer.writeheader();writer.writerows(top)
    dump(WORK/'all_node_decomposition.json',list(records.values()))
    print(json.dumps({'summary':result['summary'],'verification':result['verification'],'top20':top},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
