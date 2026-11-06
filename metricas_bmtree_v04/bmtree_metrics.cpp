#include "common.hpp"
#include "vendor/btree_set.h"
struct Node { int dim, left, right, child; };
struct Tree {
    std::array<int,2> bits;
    std::vector<Node> nodes;
    explicit Tree(const std::string& file) {
        std::ifstream f(file);
        int d, depth, id, parent;
        if (!(f >> d >> bits[0] >> bits[1] >> depth) || d!=2) { throw std::runtime_error("Arbol invalido"); }
        while (f >> id >> parent >> depth) {
            Node n;
            if (!(f >> n.dim >> n.left >> n.right >> n.child) || id!=int(nodes.size())) { throw std::runtime_error("Nodo invalido"); }
            nodes.push_back(n);
        }
    }
    uint64_t key(const Point& p) const {
        uint64_t value=0;
        std::array<int,2> used{0,0};
        int id=0, previous=-1;
        for (int consumed=0; consumed<bits[0]+bits[1]; ++consumed) {
            int d;
            if (id>=0 && nodes.at(id).dim>=0) { d=nodes.at(id).dim; }
            else {
                d=(previous+1)%2;
                while (used[d]>=bits[d]) { d=(d+1)%2; }
                id=-1;
            }
            if (d<0 || d>1 || used[d]>=bits[d]) { throw std::runtime_error("Bits invalidos"); }
            uint64_t bit=(uint64_t(p[d])>>(bits[d]-used[d]-1))&1;
            value=(value<<1)|bit;
            ++used[d]; previous=d;
            if (id>=0) {
                const auto& n=nodes.at(id);
                id=n.child>=0?n.child:(bit?n.right:n.left);
            }
        }
        return value;
    }
};
struct BMIndex {
    struct Record { Point point; bool alive; };
    Tree tree;
    btree::btree_set<std::pair<uint64_t,uint64_t>> ordered;
    std::vector<Record> records;
    explicit BMIndex(Tree t) : tree(std::move(t)) {}
    void insert(const Point& p) {
        size_t id=records.size(); records.push_back({p,true});
        ordered.insert({tree.key(p),id});
    }
    void erase(size_t id) {
        if (!records.at(id).alive) { throw std::runtime_error("ID ya eliminado"); }
        ordered.erase({tree.key(records[id].point),id}); records[id].alive=false;
    }
    std::vector<Point> enumerate(const Box& q) {
        std::vector<Point> result;
        auto end=tree.key(q.hi);
        for (auto it=ordered.lower_bound({tree.key(q.lo),0}); it!=ordered.end() && it->first<=end; ++it) {
            const auto& p=records[it->second].point;
            if (q.contains(p)) { result.push_back(p); }
        }
        return result;
    }
    size_t rekey(const Tree& next, const std::vector<Box>& regions) {
        std::vector<size_t> ids;
        for (const auto& q : regions) {
            auto end=tree.key(q.hi);
            for (auto it=ordered.lower_bound({tree.key(q.lo),0}); it!=ordered.end() && it->first<=end; ++it) {
                if (q.contains(records[it->second].point)) { ids.push_back(it->second); }
            }
        }
        std::sort(ids.begin(),ids.end()); ids.erase(std::unique(ids.begin(),ids.end()),ids.end());
        for (auto id : ids) { ordered.erase({tree.key(records[id].point),id}); }
        tree=next;
        for (auto id : ids) { ordered.insert({tree.key(records[id].point),id}); }
        // Validation is performed outside timing by the caller.
        return ids.size();
    }
    size_t owned_bytes() const {
        return sizeof(*this)+records.capacity()*sizeof(Record)+tree.nodes.capacity()*sizeof(Node)
            +ordered.bytes_used()-sizeof(ordered);
    }
};
int main(int argc,char**argv) {
    try {
        if (argc!=3) { throw std::runtime_error("Uso: bmtree_metrics DIR REPETICIONES"); }
        std::string dir=argv[1]; int repeats=std::stoi(argv[2]);
        Workload w(dir); Tree old(dir+"/old_tree.txt"), next(dir+"/new_tree.txt");
        auto start=Clock::now();
        BMIndex index(std::move(old));
        for (const auto& p:w.initial) { index.insert(p); }
        double build=elapsed(start); size_t memory_before=index.owned_bytes();
        auto vb=validate(index,w.initial,w.test); auto before=measure(index,w.test,repeats);
        start=Clock::now();
        for (auto id:w.deletions) { index.erase(id); }
        for (const auto& p:w.additions) { index.insert(p); }
        double update=elapsed(start);
        start=Clock::now(); size_t rekeyed=index.rekey(next,w.regions); double rekey=elapsed(start);
        auto va=validate(index,w.final_points,w.test); auto after=measure(index,w.test,repeats);
        emit(before,after,vb,va,build,update,rekey,w.deletions.size()+w.additions.size(),memory_before,index.owned_bytes(),
            ",\"rekeyed_points\":"+std::to_string(rekeyed));
        return (vb.mismatched || va.mismatched)?2:0;
    } catch(const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
}
