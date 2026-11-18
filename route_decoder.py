"""Graph-aware candidate Viterbi decoder."""
import numpy as np

def decode(observation_index, arc_id, probability, observations, arcs, gap_seconds=30, switch_penalty=2, disconnected_penalty=8):
    n=len(observations); output=np.full(n,-1,dtype=np.int64)
    groups=[np.flatnonzero(observation_index==i) for i in range(n)]
    start=0
    while start<n:
        if not len(groups[start]): start+=1; continue
        end=start+1
        while end<n and len(groups[end]) and float(observations.iloc[end].dt_s)<=gap_seconds: end+=1
        scores=[]; backs=[]
        for i in range(start,end):
            ids=groups[i]; emission=np.log(np.clip(probability[ids],1e-8,1))
            if i==start: scores.append(emission); backs.append(None); continue
            previous=arc_id[groups[i-1]]; current=arc_id[ids]
            same=previous[:,None]==current[None,:]
            connected=arcs[previous,1][:,None]==arcs[current,0][None,:]
            transition=np.where(same,0,np.where(connected,-switch_penalty,-disconnected_penalty))
            table=scores[-1][:,None]+transition
            back=np.argmax(table,axis=0)
            scores.append(emission+table[back,np.arange(len(ids))]); backs.append(back)
        state=int(np.argmax(scores[-1]))
        for i in range(end-1,start-1,-1):
            output[i]=int(arc_id[groups[i][state]])
            if i>start: state=int(backs[i-start][state])
        start=end
    return output

