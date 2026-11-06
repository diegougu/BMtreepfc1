#define main benchmark_entry_unused
#include "bmtree_metrics.cpp"
#undef main
struct Fake {
    std::vector<Point> reply;
    std::vector<Point> enumerate(const Box&) { return reply; }
};
int main(int argc,char**argv) {
    if (argc==1) {
        std::vector<Point> truth{{0,0},{1,1}};
        std::vector<Box> qs{{{0,0},{2,2}}};
        Fake missing{{{0,0}}}; auto a=validate(missing,truth,qs);
        Fake extra{{{0,0},{1,1},{2,2}}}; auto b=validate(extra,truth,qs);
        Fake wrong{{{7,7}}}; auto c=validate(wrong,truth,qs);
        Fake empty{{}}; auto d=validate(empty,{},qs);
        if (a.recall()!=0.5 || a.mismatched!=1 || b.recall()!=1 || b.false_positives!=1 || b.mismatched!=1 || c.recall()!=0 || c.false_positives!=1 || d.recall()!=1 || d.empty_queries!=1) { return 2; }
        std::cout<<"missing=0.5 extra=1(with false positive detected) wrong=0 empty=1(convention)\n";
        return 0;
    }
    std::string dir=argv[1]; Workload w(dir);
    Tree old(dir+"/old_tree.txt"), next(dir+"/new_tree.txt");
    BMIndex index(old);
    for (auto p:w.initial) { index.insert(p); }
    auto before=validate(index,w.initial,w.test);
    for (auto id:w.deletions) { index.erase(id); }
    for (auto p:w.additions) { index.insert(p); }
    index.rekey(next,w.regions);
    for (auto entry:index.ordered) {
        if (entry.first!=next.key(index.records[entry.second].point)) { return 3; }
    }
    // Dump actual answers; Python checks them with its own oracle.
    for (auto q:w.test) {
        auto result=index.enumerate(q); std::sort(result.begin(),result.end());
        std::cout<<result.size();
        for (auto p:result) { std::cout<<' '<<p[0]<<' '<<p[1]; }
        std::cout<<'\n';
    }
    return before.mismatched?4:0;
}
