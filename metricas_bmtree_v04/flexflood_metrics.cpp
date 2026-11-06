#include "common.hpp"
#include "flex/src/FlexFlood.hpp"
struct FlexAdapter {
    FlexFlood<double,2>& index;
    std::vector<Point> enumerate(const Box& q) { return index.enumerate(q.lo,q.hi); }
};
int main(int argc,char**argv) {
    try {
        if (argc!=6) { throw std::runtime_error("Uso: flexflood_metrics DIR REPETICIONES SEGUNDOS_RF MUESTRAS SEMILLA"); }
        Workload w(argv[1]); int repeats=std::stoi(argv[2]);
        double learning=std::stod(argv[3]); int samples=std::stoi(argv[4]);
        rng.mt.seed(std::stoull(argv[5]));
        std::vector<Point> lower, upper;
        for (const auto& q:w.training) { lower.push_back(q.lo); upper.push_back(q.hi); }
        auto start=Clock::now();
        FlexFlood<double,2> index(w.initial.begin(),w.initial.end(),lower,upper,samples,learning);
        double build=elapsed(start); size_t memory_before=index.owned_bytes();
        FlexAdapter adapter{index};
        auto vb=validate(adapter,w.initial,w.test); auto before=measure(adapter,w.test,repeats);
        start=Clock::now();
        for (auto id:w.deletions) { index.erase(w.initial[id]); }
        for (const auto& p:w.additions) { index.insert(p); }
        double update=elapsed(start);
        auto va=validate(adapter,w.final_points,w.test); auto after=measure(adapter,w.test,repeats);
        emit(before,after,vb,va,build,update,index.maintenance_seconds,w.deletions.size()+w.additions.size(),memory_before,index.owned_bytes(),
            ",\"splits\":"+std::to_string(index.splits)+",\"merges\":"+std::to_string(index.merges)+",\"equalizes\":"+std::to_string(index.equalizes));
        index.clear();
        return (vb.mismatched || va.mismatched)?2:0;
    } catch(const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
}
