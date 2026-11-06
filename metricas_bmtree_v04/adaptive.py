"""Independent implementation of Sections V–VI; not authors' released code.
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

    boxes: tuple = ()

    def domains(self):
        return self.boxes or ((self.lo, self.hi),)

    def mask(self, points):
        result = np.zeros(len(points), dtype=bool)
        for lo, hi in self.domains():
            result |= np.all((points >= lo) & (points < np.asarray(hi)+1), axis=1)
        return result

    def ratio(self, bits):
        return sum(math.prod((h-l+1)/(1 << b) for l,h,b in zip(lo,hi,bits))
                   for lo,hi in self.domains())


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


def child_regions(tree, r, bits):
    n = get_node(tree, r.path)
    if n is None or n.dim < 0:
        return []
    d = n.dim
    used = list(r.used)
    width = 1 << (bits[d]-used[d]-1)
    used[d] += 1
    if n.child is not None:
        return [Region(r.path+('child',), r.lo, r.hi, tuple(used), d, r.domains())]
    result = []
    for bit, edge in enumerate(('left', 'right')):
        if getattr(n, edge) is None:
            continue
        boxes = []
        # A skipped/shared high bit makes a descendant a UNION of stripes.
        for lo, hi in r.domains():
            for stripe in range(lo[d]//width, hi[d]//width+1):
                if stripe % 2 != bit:
                    continue
                l, h = list(lo), list(hi)
                l[d], h[d] = max(l[d],stripe*width), min(h[d],(stripe+1)*width-1)
                boxes.append((tuple(l),tuple(h)))
        if boxes:
            lo=tuple(min(x[0][i] for x in boxes) for i in range(2))
            hi=tuple(max(x[1][i] for x in boxes) for i in range(2))
            result.append(Region(r.path+(edge,),lo,hi,tuple(used),d,tuple(boxes)))
    return result


def regions(tree,bits,max_depth):
    queue=[Region((),(0,0),tuple((1<<b)-1 for b in bits),(0,0),-1)]
    for r in queue:
        if len(r.path)>=max_depth:
            continue
        yield r
        queue.extend(child_regions(tree,r,bits))


def cells(tree,region,bits):
    # Actual grandchildren, keeping shared children shared (Section VI-A).
    pending=[region]
    for _ in range(2):
        pending=[c for r in pending for c in (child_regions(tree,r,bits) or [r])]
    return pending


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
    sr_o=scan_ranges(tree,old,qo,bits,block_size)
    sr_n=scan_ranges(tree,new,qn,bits,block_size)
    candidates=[]
    passed=set()
    for r in regions(tree,bits,max_depth):
        if r.path and r.path[:-1] not in passed:
            continue
        cs=cells(tree,r,bits)
        shift_d=js([c.mask(old).sum() for c in cs],[c.mask(new).sum() for c in cs])
        query_shifts=[]
        for c in cs:
            labels_o,labels_n=query_clusters(qo[c.mask(co)],qn[c.mask(cn)])
            query_shifts.append(js(np.bincount(labels_o,minlength=4),np.bincount(labels_n,minlength=4)))
        shift_q=float(np.mean(query_shifts))
        shift=alpha*shift_d+(1-alpha)*shift_q
        mo=r.mask(co); mn=r.mask(cn)
        op=(float(sr_n[mn].mean()) if mn.any() else 0)-(float(sr_o[mo].mean()) if mo.any() else 0)
        if shift>=threshold:
            passed.add(r.path)
            candidates.append((len(r.path),-op,r,shift))
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
    children: dict
    visits: int = 0
    value: float = -math.inf
    action: tuple = ()
    greedy_done: bool = False

    def observe(self, reward):
        self.visits += 1
        self.value = max(self.value, reward)


def uct(child, parent_visits, c=1.0):
    if child.visits == 0:
        return math.inf
    # Literal printed expression in Section V, p. 8 (V stores a maximum).
    return child.value/child.visits + c*math.sqrt(math.log(max(parent_visits,1))/child.visits)


def choices(used, bits):
    return tuple((d, split) for d in range(2) if used[d]<bits[d]
                 for split in ((1,0) if sum(used)+1<sum(bits) else (0,)))


def fill_one(tree, path, used, action, bits):
    d, split = action
    u=list(used); u[d]+=1; u=tuple(u)
    if sum(u)==sum(bits):
        node=Node(d); frontier=[]
    elif split:
        node=Node(d,Node(),Node())
        frontier=[(path+('left',),u),(path+('right',),u)]
    else:
        node=Node(d,child=Node()); frontier=[(path+('child',),u)]
    return replace(tree,path,node),frontier


def greedy_action(tree,frontier,bits,cost):
    actions=[]
    for path,used in frontier:
        options=choices(used,bits)
        # Ties: lowest dimension, then split. No forced split constraint.
        action=min(options,key=lambda a:cost(fill_one(tree,path,used,a,bits)[0]))
        tree,_=fill_one(tree,path,used,action,bits)
        actions.append(action)
    return tuple(actions)


def train_forest(tree,bits,selected,data,queries,rng,levels=3,rollouts=32,
                 branch_cap=None,split_depth=None,block_size=50):
    """GAS, joint dimension/split actions, maximum backup, UCT and Z completion.
    Deprecated branch_cap/split_depth arguments are rejected, never silently used.
    """
    if branch_cap is not None or split_depth is not None:
        raise ValueError('v04 removes branch_cap and forced split_depth; use levels/rollouts')
    if levels<1 or rollouts<1 or block_size<1:
        raise ValueError('Positive levels, rollouts and block_size required')
    centers=(queries[:,:2]+queries[:,2:])/2
    mask=np.zeros(len(queries),dtype=bool)
    for r in selected:
        mask |= r.mask(centers)
    queries=queries[mask]
    if not len(queries) or not selected:
        return tree, {'evaluations':0,'reason':'no selected training queries'}
    original=tree
    all_points=np.concatenate([data,queries[:,:2],queries[:,2:]])
    fixed_keys=encode(original,all_points,bits)
    changed=np.zeros(len(all_points),dtype=bool)
    for r in selected:
        changed |= r.mask(all_points)
    @lru_cache(maxsize=4096)
    def cost(t):
        keys=fixed_keys.copy()
        keys[changed]=encode(t,all_points[changed],bits)
        starts=np.sort(keys[:len(data)])[::block_size]
        if not len(starts):
            return 0.0
        lower,upper=np.split(keys[len(data):],2)
        a=np.clip(np.searchsorted(starts,lower,'right')-1,0,len(starts)-1)
        b=np.clip(np.searchsorted(starts,upper,'right')-1,0,len(starts)-1)
        return float(np.mean(b-a))
    initial_cost=cost(original)
    for r in selected:
        tree=replace(tree,r.path,Node())
    frontier=tuple((r.path,r.used) for r in selected if sum(r.used)<sum(bits))
    z_cost=float(scan_ranges(Node(0),data,queries,bits,block_size).mean())
    normalizer=z_cost if z_cost>0 else 1.0
    gas_calls=0; expansions=0; deepest=0
    root=Policy(tree,frontier,{})
    def transition(p, action):
        t=p.tree; f=[]
        for (path,used),a in zip(p.frontier,action):
            t,nxt=fill_one(t,path,used,a,bits); f.extend(nxt)
        return Policy(t,tuple(f),{},action=action)
    def unseen(p):
        nonlocal gas_calls
        if not p.greedy_done:
            p.greedy_done=True; gas_calls+=1
            a=greedy_action(p.tree,p.frontier,bits,cost)
            if a not in p.children:
                return a
        opts=[choices(u,bits) for _,u in p.frontier]
        if len(p.children)==math.prod(map(len,opts)):
            return None
        # Lazy sampling avoids materializing the exponential joint action space.
        for _ in range(32):
            a=tuple(o[int(rng.integers(len(o)))] for o in opts)
            if a not in p.children:
                return a
        # Guaranteed progress near exhaustion; no action-space truncation.
        return next(a for a in itertools.product(*opts) if a not in p.children)
    for step in range(levels):
        if not root.frontier:
            break
        for _ in range(rollouts):
            p=root; route=[p]; depth=step
            while p.frontier and depth<levels:
                a=unseen(p)
                if a is not None:
                    child=transition(p,a);p.children[a]=child
                    p=child;route.append(p);depth+=1;expansions+=1
                    break
                p=max(p.children.values(),key=lambda ch:uct(ch,route[-1].visits))
                route.append(p);depth+=1
            deepest=max(deepest,depth)
            reward=(z_cost-cost(p.tree))/normalizer
            for state in route:
                state.observe(reward)
        if not root.children:
            break
        root=max(root.children.values(),key=lambda ch:ch.value)
    final_cost=cost(root.tree)
    return root.tree, {'evaluations':cost.cache_info().misses,'cache_hits':cost.cache_info().hits,
        'sr_before':initial_cost,'sr_after':final_cost,'sr_z_baseline':z_cost,
        'gas_calls':gas_calls,'expanded_states':expansions,'deepest_level':deepest,
        'backup':'maximum','uct':'V_max/visits + sqrt(log(parent_visits)/visits)',
        'levels':levels,'rollouts_per_level':rollouts,'joint_action_space_truncated':False}


def adapt(tree,bits,old,new,qo,qn,seed=42,threshold=.1,budget=.5,max_depth=5,
          alpha=.5,block_size=50,levels=3,rollouts=32,split_depth=None,max_rounds=None):
    if not 0<budget<=1 or not 0<=alpha<=1 or threshold<0:
        raise ValueError('Invalid shift/budget parameters')
    if max_rounds is not None and max_rounds < 1:
        raise ValueError('max_rounds must be positive or None')
    begin=time.perf_counter()
    candidates=detect(tree,bits,old,new,qo,qn,threshold,budget,max_depth,alpha,block_size)
    selected=select(candidates,bits,budget)
    detection=time.perf_counter()-begin
    begin=time.perf_counter(); result=tree; reports=[]; rng=np.random.default_rng(seed)
    pending=selected; used=list(selected); training=0.0
    # Algorithm 2 trains ALL nodes returned by Algorithm 1 jointly first.
    # On insufficient gain, select untouched regions of the new tree, keeping
    # a cumulative strict area budget and never retraining an already used path.
    while pending:
        if max_rounds is not None and len(reports)>=max_rounds:
            break
        train_start=time.perf_counter()
        result,report=train_forest(result,bits,pending,new,qn,rng,levels,rollouts,
                                  split_depth=split_depth,block_size=block_size)
        training+=time.perf_counter()-train_start
        reports.append(report)
        gain=(report.get('sr_before',0)-report.get('sr_after',0))/max(report.get('sr_before',0),1e-12)
        if gain>=.01 or (max_rounds is not None and len(reports)>=max_rounds):
            break
        detect_start=time.perf_counter()
        more=detect(result,bits,old,new,qo,qn,threshold,budget,max_depth,alpha,block_size)
        extended=select(more,bits,budget,already=used)
        detection+=time.perf_counter()-detect_start
        pending=extended[len(used):]
        used=extended
    return result,used, {'detection_seconds':detection,
        'training_seconds':training,
        'selected_paths':['/'.join(r.path) or 'root' for r in used],
        'space_ratio':sum(r.ratio(bits) for r in used),'candidates':len(candidates),'rounds':reports,
        'automatic_extra_rounds_implemented':True,'round_limit':max_rounds,
        'eligible_regions':len(selected),'retraining_performed':bool(reports),
        'reason':'trained selected regions' if reports else 'no feasible shifted region under threshold and area budget'}
