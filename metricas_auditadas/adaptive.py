"""Independent, auditable Section VI reconstruction; not authors' released code.
Integer 2D points, inclusive range boxes, <= 62 bits total.
"""
from dataclasses import dataclass
from functools import lru_cache
import itertools
import math
import time
import numpy as np

@dataclass(frozen=True)
class Node:
    dim: int = -1
    left: object = None
    right: object = None
    child: object = None

@dataclass(frozen=True)
class Region:
    path: tuple
    lo: tuple
    hi: tuple
    used: tuple
    parent_dim: int

    def mask(self, points):
        # Integer lattice cells have half-open geometric extent [lo, hi+1).
        # Query centers may be half-integers; <=hi incorrectly dropped centers
        # such as 3.5 between neighboring cells ending at 3 and starting at 4.
        return np.all((points >= self.lo) & (points < np.asarray(self.hi)+1), axis=1)

    def ratio(self, bits):
        return math.prod((h-l+1)/(1 << b) for l,h,b in zip(self.lo,self.hi,bits))


def read_tree(path):
    with open(path) as f:
        lines = [list(map(int, s.split())) for s in f if s.strip()]
    bits = tuple(lines[0][1:])
    if lines[0][0] != 2 or len(bits) != 2 or min(bits)<1 or max(bits)>31:
        raise ValueError('Only 2D, positive bits and sum(bits)<=62 supported')
    rows = {r[0]: r for r in lines[2:]}
    visiting = set()
    def build(i):
        if i == -1:
            return None
        if i in visiting:
            raise ValueError('Cycle in input tree')
        visiting.add(i)
        r=rows[i]
        if len(r) != 7:
            raise ValueError('Expected seven columns per node')
        node=Node(r[3], build(r[4]), build(r[5]), build(r[6]))
        visiting.remove(i)
        return node
    tree=build(0)
    # Validate each path; a shared child consumes the same dimension on both branches.
    def validate(n, used):
        if n is None or n.dim == -1: return
        if not 0 <= n.dim < 2 or used[n.dim] >= bits[n.dim]:
            raise ValueError('Invalid/exhausted bit dimension')
        u=list(used); u[n.dim]+=1
        if n.child is not None and (n.left is not None or n.right is not None):
            raise ValueError('Ambiguous shared/split child')
        for ch in (n.left,n.right,n.child): validate(ch,tuple(u))
    validate(tree,(0,0))
    return tree,bits


def write_tree(path, tree, bits):
    queue=[(tree,-1,0)]; rows=[]
    while len(rows)<len(queue):
        i=len(rows); n,parent,depth=queue[i]; links=[]
        for ch in (n.left,n.right,n.child):
            links.append(-1 if ch is None else len(queue))
            if ch is not None: queue.append((ch,i,depth+1))
        rows.append((i,parent,depth,n.dim,*links))
    with open(path,'w') as f:
        f.write(f'2 {bits[0]} {bits[1]}\n{max(r[2] for r in rows)}\n')
        for row in rows: f.write(' '.join(map(str,row))+'\n')


def encode(tree, points, bits):
    p=np.asarray(points,dtype=np.int64).reshape(-1,2)
    out=np.zeros(len(p),dtype=np.int64)
    def visit(n, ids, used, previous):
        if not len(ids): return
        if n is None or n.dim<0:
            u=list(used); dim=previous
            while sum(u)<sum(bits):
                dim=(dim+1)%2
                if u[dim]==bits[dim]: continue
                shift=bits[dim]-u[dim]-1
                out[ids]=(out[ids]<<1)|((p[ids,dim]>>shift)&1)
                u[dim]+=1
            return
        d=n.dim; u=list(used)
        bit=(p[ids,d]>>(bits[d]-u[d]-1))&1
        out[ids]=(out[ids]<<1)|bit
        u[d]+=1
        if n.child is not None:
            visit(n.child,ids,tuple(u),d)
        else:
            visit(n.left,ids[bit==0],tuple(u),d)
            visit(n.right,ids[bit==1],tuple(u),d)
    visit(tree,np.arange(len(p)),(0,0),-1)
    return out


def get_node(tree,path):
    for edge in path: tree=getattr(tree,edge)
    return tree


def replace(tree,path,node):
    if not path: return node
    edge=path[0]
    children={x:getattr(tree,x) for x in ('left','right','child')}
    children[edge]=replace(children[edge],path[1:],node)
    return Node(tree.dim,**children)


def regions(tree,bits,max_depth):
    root=Region((),(0,0),tuple((1<<b)-1 for b in bits),(0,0),-1)
    queue=[root]
    for r in queue:
        if len(r.path)>=max_depth: continue
        yield r
        n=get_node(tree,r.path)
        # Below shared children the spatial domain is a union. Do not pretend
        # that it is a single box: the containing region may still be retrained.
        if n is None or n.dim<0 or n.child is not None: continue
        d=n.dim; middle=(r.lo[d]+r.hi[d])//2
        u=list(r.used); u[d]+=1
        for edge in ('left','right'):
            if getattr(n,edge) is None: continue
            lo=list(r.lo); hi=list(r.hi)
            if edge=='left': hi[d]=middle
            else: lo[d]=middle+1
            queue.append(Region(r.path+(edge,),tuple(lo),tuple(hi),tuple(u),d))


def cells(tree,region,bits):
    """Two consumed bits below N; implicit Z completion for shallow nodes."""
    pending=[(get_node(tree,region.path),region.lo,region.hi,region.used,region.parent_dim)]
    for _ in range(2):
        nxt=[]
        for n,lo,hi,used,prev in pending:
            if sum(used)==sum(bits):
                nxt.append((n,lo,hi,used,prev)); continue
            d=n.dim if n is not None and n.dim>=0 else (prev+1)%2
            while used[d]==bits[d]: d=(d+1)%2
            u=list(used); u[d]+=1; mid=(lo[d]+hi[d])//2
            for j in range(2):
                l=list(lo); h=list(hi)
                if j: l[d]=mid+1
                else: h[d]=mid
                ch=None if n is None else (n.child if n.child is not None else (n.left if not j else n.right))
                nxt.append((ch,tuple(l),tuple(h),tuple(u),d))
        pending=nxt
    return [Region((),lo,hi,u,p) for _,lo,hi,u,p in pending]


def js(a,b):
    # Explicit absent-observation category makes empty/nonempty windows defined.
    a=np.append(np.asarray(a,dtype=float),0.0)
    b=np.append(np.asarray(b,dtype=float),0.0)
    if a.sum()==0: a[-1]=1
    if b.sum()==0: b[-1]=1
    a/=a.sum(); b/=b.sum(); m=(a+b)/2
    def kl(x):
        mask=x>0
        return float(np.sum(x[mask]*np.log2(x[mask]/m[mask])))
    return (kl(a)+kl(b))/2


def query_clusters(old,new,k=4):
    """Deterministic joint clustering on log(area), log(aspect), training only."""
    q=np.concatenate([old,new]); lengths=q[:,2:]-q[:,:2]+1
    features=np.column_stack([np.log(lengths.prod(axis=1)),np.log(lengths[:,0]/lengths[:,1])])
    if not len(q): return np.array([],dtype=int),np.array([],dtype=int)
    features=(features-features.mean(axis=0))/np.maximum(features.std(axis=0),1e-12)
    centers=[features[0]]
    for _ in range(min(k,len(q))-1):
        dist=np.min(((features[:,None,:]-np.array(centers))**2).sum(axis=2),axis=1)
        centers.append(features[int(np.argmax(dist))])
    centers=np.array(centers)
    for _ in range(30):
        labels=np.argmin(((features[:,None,:]-centers)**2).sum(axis=2),axis=1)
        updated=np.array([features[labels==i].mean(axis=0) if np.any(labels==i) else c for i,c in enumerate(centers)])
        if np.allclose(updated,centers): break
        centers=updated
    return labels[:len(old)],labels[len(old):]


def scan_ranges(tree,data,queries,bits,block_size):
    if len(data)==0: return np.zeros(len(queries))
    starts=np.sort(encode(tree,data,bits))[::block_size]
    lower=encode(tree,queries[:,:2],bits); upper=encode(tree,queries[:,2:],bits)
    a=np.clip(np.searchsorted(starts,lower,side='right')-1,0,len(starts)-1)
    b=np.clip(np.searchsorted(starts,upper,side='right')-1,0,len(starts)-1)
    return (b-a).astype(float)  # Eq. 2: block difference, not elapsed query time.


def detect(tree,bits,old,new,qo,qn,threshold=.1,budget=.5,max_depth=5,alpha=.5,block_size=50):
    co=(qo[:,:2]+qo[:,2:])/2; cn=(qn[:,:2]+qn[:,2:])/2
    labels_o,labels_n=query_clusters(qo,qn)
    sr_o=scan_ranges(tree,old,qo,bits,block_size)
    sr_n=scan_ranges(tree,new,qn,bits,block_size)
    candidates=[]
    for r in regions(tree,bits,max_depth):
        cs=cells(tree,r,bits)
        shift_d=js([c.mask(old).sum() for c in cs],[c.mask(new).sum() for c in cs])
        shift_q=np.mean([js(np.bincount(labels_o[c.mask(co)],minlength=4),np.bincount(labels_n[c.mask(cn)],minlength=4)) for c in cs])
        shift=alpha*shift_d+(1-alpha)*shift_q
        mo=r.mask(co); mn=r.mask(cn)
        op=(float(sr_n[mn].mean()) if mn.any() else 0)-(float(sr_o[mo].mean()) if mo.any() else 0)
        if shift>=threshold and mn.any(): candidates.append((len(r.path),-op,r,shift))
    # BFS priority, descending OP within level; disjoint boxes, strict area budget.
    candidates.sort(key=lambda x:(x[0],x[1],x[2].path))
    return candidates


def select(candidates,bits,budget,already=()):
    chosen=list(already); area=sum(r.ratio(bits) for r in chosen)
    for _,_,r,_ in candidates:
        if any(r.path[:len(s.path)]==s.path or s.path[:len(r.path)]==r.path for s in chosen): continue
        if area+r.ratio(bits)<budget-1e-12:
            chosen.append(r); area+=r.ratio(bits)
    return chosen


@dataclass
class Policy:
    tree: Node
    frontier: tuple
    children: list
    visits: int=0
    total: float=0.0
    action: tuple=()


def train_forest(tree,bits,selected,data,queries,rng,levels=3,rollouts=32,branch_cap=16,split_depth=6,block_size=50):
    """Joint dimension/split actions on a forest, UCT, Z-completion simulations.
    Bounded sampled action expansion; best observed training-SR incumbent.
    """
    centers=(queries[:,:2]+queries[:,2:])/2
    mask=np.zeros(len(queries),dtype=bool)
    for r in selected: mask|=r.mask(centers)
    queries=queries[mask]
    if not len(queries) or not selected: return tree, {'evaluations':0}
    original=tree
    # Only re-encode changed domains: reuse all other sampled data/query keys.
    all_points=np.concatenate([data,queries[:,:2],queries[:,2:]])
    fixed_keys=encode(original,all_points,bits)
    changed=np.zeros(len(all_points),dtype=bool)
    for r in selected: changed|=r.mask(all_points)
    @lru_cache(maxsize=4096)
    def cost(t):
        keys=fixed_keys.copy(); keys[changed]=encode(t,all_points[changed],bits)
        starts=np.sort(keys[:len(data)])[::block_size]
        if not len(starts): return 0.0
        qkeys=keys[len(data):]; a,b=np.split(qkeys,2)
        lower=np.clip(np.searchsorted(starts,a,'right')-1,0,len(starts)-1)
        upper=np.clip(np.searchsorted(starts,b,'right')-1,0,len(starts)-1)
        return float(np.mean(upper-lower))
    initial_cost=cost(original); best_tree=original; best_cost=initial_cost
    for r in selected: tree=replace(tree,r.path,Node())
    frontier=tuple((r.path,r.used) for r in selected)
    z_cost=float(scan_ranges(Node(0),data,queries,bits,block_size).mean())
    normalizer=max(z_cost,1.)
    def transition(p):
        t=p.tree; next_front=[]; action=[]
        for path,used in p.frontier:
            valid=[d for d in range(2) if used[d]<bits[d]]
            if not valid: continue
            d=int(rng.choice(valid))
            # Preserve a shallow spatial partition so that later partial
            # adaptation has independently selectable subspaces.
            split=1 if sum(used)<2 else (int(rng.integers(2)) if sum(used)<split_depth else 0)
            u=list(used); u[d]+=1; action.append((d,split))
            if sum(u)==sum(bits): node=Node(d)
            elif split:
                node=Node(d,Node(),Node()); next_front.extend([(path+('left',),tuple(u)),(path+('right',),tuple(u))])
            else:
                node=Node(d,child=Node()); next_front.append((path+('child',),tuple(u)))
            t=replace(t,path,node)
        return Policy(t,tuple(next_front),[],action=tuple(action))
    # Receding horizon: commit an action, then search again from the new frontier.
    for step in range(levels):
        root=Policy(tree,frontier,[])
        for _ in range(rollouts):
            p=root; route=[p]
            for _depth in range(min(3,levels-step)):
                if not p.frontier: break
                child=None
                if len(p.children)<branch_cap:
                    for _attempt in range(8):
                        c=transition(p)
                        if all(c.action!=x.action for x in p.children):
                            p.children.append(c); child=c; break
                if child is None:
                    if not p.children: break
                    child=max(p.children,key=lambda c:c.total/max(1,c.visits)+math.sqrt(math.log(p.visits+2)/max(1,c.visits)))
                p=child; route.append(p)
            score=cost(p.tree)
            if score<best_cost: best_tree,best_cost=p.tree,score
            reward=(z_cost-score)/normalizer
            for state in route: state.visits+=1; state.total+=reward
        if not root.children: break
        winner=max(root.children,key=lambda c:c.total/max(c.visits,1))
        tree,frontier=winner.tree,winner.frontier
        if not frontier: break
    return best_tree, {'evaluations':cost.cache_info().misses,'sr_before':initial_cost,'sr_after':best_cost,'sr_z_baseline':z_cost}


def adapt(tree,bits,old,new,qo,qn,seed=42,threshold=.1,budget=.5,max_depth=5,alpha=.5,block_size=50,levels=3,rollouts=32,split_depth=6,max_rounds=8):
    begin=time.perf_counter()
    candidates=detect(tree,bits,old,new,qo,qn,threshold,budget,max_depth,alpha,block_size)
    eligible=select(candidates,bits,budget)
    detection=time.perf_counter()-begin
    begin=time.perf_counter(); result=tree; reports=[]; rng=np.random.default_rng(seed)
    selected=[]
    # Begin with the highest-priority feasible region; add untouched regions
    # when the measured training gain is below 1%, under a cumulative budget.
    for region in eligible[:max_rounds]:
        selected.append(region)
        result,report=train_forest(result,bits,selected,new,qn,rng,levels,rollouts,split_depth=split_depth,block_size=block_size)
        reports.append(report)
        gain=(report.get('sr_before',0)-report.get('sr_after',0))/max(report.get('sr_before',0),1e-12)
        if gain>=.01: break
    return result, selected, {'detection_seconds':detection,'training_seconds':time.perf_counter()-begin,
        'selected_paths':['/'.join(r.path) or 'root' for r in selected],
        'space_ratio':sum(r.ratio(bits) for r in selected),'candidates':len(candidates),'rounds':reports,
        'automatic_extra_rounds_implemented':True,
        'round_limit':max_rounds,'eligible_regions':len(eligible)}
