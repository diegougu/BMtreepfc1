// Standalone RAM benchmark. BMTree mapping + cpp-btree, no database.
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>
#include "vendor/btree_set.h"
using Clock=std::chrono::steady_clock;
using Point=std::array<uint32_t,2>;
using Key=std::pair<uint64_t,uint64_t>;
struct Node {int dim,left,right,child;};
struct Tree {
    std::array<int,2> bits;
    std::vector<Node> nodes;
    explicit Tree(const std::string& path) {
        std::ifstream f(path); int d,depth;
        if(!(f>>d>>bits[0]>>bits[1]>>depth)||d!=2||bits[0]<1||bits[1]<1||bits[0]>31||bits[1]>31)
            throw std::runtime_error("invalid tree header");
        int id,parent;
        while(f>>id>>parent>>depth) {Node n;if(!(f>>n.dim>>n.left>>n.right>>n.child)||id!=int(nodes.size()))throw std::runtime_error("expected dense BFS node IDs");nodes.push_back(n);}
        if(nodes.empty())throw std::runtime_error("missing root");
    }
    uint64_t key(Point p) const {
        uint64_t value=0; std::array<int,2> used{0,0};int id=0,previous=-1;
        for(int consumed=0;consumed<bits[0]+bits[1];++consumed) {
            int dim;
            if(id>=0 && nodes.at(id).dim>=0)dim=nodes.at(id).dim;
            else {dim=(previous+1)%2;while(used[dim]>=bits[dim])dim=(dim+1)%2;id=-1;}
            if(dim<0||dim>1||used[dim]>=bits[dim])throw std::runtime_error("invalid bit path");
            uint64_t bit=(p[dim]>>(bits[dim]-used[dim]-1))&1;
            value=(value<<1)|bit;++used[dim];previous=dim;
            if(id>=0){const auto& n=nodes.at(id);id=n.child>=0?n.child:(bit?n.right:n.left);}
        }
        return value;
    }
};
struct Box {Point lo,hi;bool contains(Point p)const{return p[0]>=lo[0]&&p[0]<=hi[0]&&p[1]>=lo[1]&&p[1]<=hi[1];}};
struct Record{Point p;bool alive=true;};
static double seconds(Clock::time_point t){return std::chrono::duration<double>(Clock::now()-t).count();}
std::vector<Point> read_points(const std::string& p,const Tree&t){
    std::ifstream f(p);if(!f)throw std::runtime_error("cannot read "+p);
    std::vector<Point> a;uint64_t x,y;
    while(f>>x>>y){if(x>=(1ULL<<t.bits[0])||y>=(1ULL<<t.bits[1]))throw std::runtime_error("coordinate out of domain");a.push_back({uint32_t(x),uint32_t(y)});}
    if(!f.eof())throw std::runtime_error("malformed points");return a;
}
std::vector<Box> read_boxes(const std::string&p,const Tree&t){
    std::ifstream f(p);if(!f)throw std::runtime_error("cannot read "+p);
    std::vector<Box>a;uint64_t x,y,X,Y;
    while(f>>x>>y>>X>>Y){if(x>X||y>Y||X>=(1ULL<<t.bits[0])||Y>=(1ULL<<t.bits[1]))throw std::runtime_error("invalid query");a.push_back({{uint32_t(x),uint32_t(y)},{uint32_t(X),uint32_t(Y)}});}
    if(!f.eof())throw std::runtime_error("malformed boxes");return a;
}
struct Index {
    Tree tree;
    btree::btree_set<Key> index;
    std::vector<Record> records;
    explicit Index(Tree t):tree(std::move(t)){}
    void insert(Point p){uint64_t id=records.size();records.push_back({p,true});index.insert({tree.key(p),id});}
    void erase(uint64_t id){if(id>=records.size()||!records[id].alive)throw std::runtime_error("invalid erase");index.erase({tree.key(records[id].p),id});records[id].alive=false;}
    void query(const Box&q,std::vector<uint64_t>& result)const{
        result.clear();uint64_t a=tree.key(q.lo),b=tree.key(q.hi);
        for(auto it=index.lower_bound({a,0});it!=index.end()&&it->first<=b;++it)
            if(q.contains(records[it->second].p))result.push_back(it->second);
    }
    size_t rekey(const Tree&next,const std::vector<Box>&regions){
        std::vector<uint64_t>ids;
        // The selected boxes are disjoint prefix regions. Collect IDs before
        // mutating: cpp-btree invalidates iterators during erase/insert.
        for(const auto&r:regions){auto a=tree.key(r.lo),b=tree.key(r.hi);
            for(auto it=index.lower_bound({a,0});it!=index.end()&&it->first<=b;++it)
                if(r.contains(records[it->second].p))ids.push_back(it->second);
        }
        std::sort(ids.begin(),ids.end());ids.erase(std::unique(ids.begin(),ids.end()),ids.end());
        for(auto id:ids)index.erase({tree.key(records[id].p),id});
        tree=next;
        for(auto id:ids)index.insert({tree.key(records[id].p),id});
        return ids.size();
    }
    void verify(const std::vector<Box>&queries)const{
        // Out of timed regions; compare full IDs, including duplicate points.
        size_t alive=0;for(const auto&r:records)alive+=r.alive;
        if(alive!=index.size())throw std::runtime_error("index cardinality mismatch");
        for(const auto&r:records)if(r.alive && (r.p[0]>=(1ULL<<tree.bits[0])))throw std::runtime_error("bad point");
        for(const auto&entry:index)if(!records[entry.second].alive||tree.key(records[entry.second].p)!=entry.first)throw std::runtime_error("stale key after adaptation");
        std::vector<uint64_t> actual,expected;
        for(const auto&q:queries){query(q,actual);expected.clear();
            for(size_t i=0;i<records.size();++i)if(records[i].alive&&q.contains(records[i].p))expected.push_back(i);
            std::sort(actual.begin(),actual.end());if(actual!=expected)throw std::runtime_error("range result differs from brute force");
        }
    }
};
struct Metrics {double latency,qps;uint64_t checksum;};
Metrics measure(const Index&idx,const std::vector<Box>&queries,int repeats){
    std::vector<uint64_t> out;uint64_t checksum=0;
    for(const auto&q:queries)idx.query(q,out); // warmup also reserves result storage
    auto start=Clock::now();
    for(int r=0;r<repeats;++r)for(const auto&q:queries){idx.query(q,out);checksum+=out.size();}
    double elapsed=seconds(start),count=double(repeats)*queries.size();
    return {elapsed*1e6/count,count/elapsed,checksum};
}
int main(int argc,char**argv){try{
    if(argc!=3)throw std::runtime_error("usage: benchmark RUN_DIRECTORY REPEATS");
    std::string dir=argv[1];int repeats=std::stoi(argv[2]);if(repeats<1)throw std::runtime_error("invalid repeats");
    Tree old(dir+"/old_tree.txt"),next(dir+"/new_tree.txt");if(old.bits!=next.bits)throw std::runtime_error("changed bit domain");
    auto points=read_points(dir+"/points.tsv",old),added=read_points(dir+"/insertions.tsv",old);
    auto queries=read_boxes(dir+"/test_queries.tsv",old),regions=read_boxes(dir+"/regions.tsv",old);
    if(queries.empty())throw std::runtime_error("no test queries");
    std::ifstream f(dir+"/deletions.tsv");if(!f)throw std::runtime_error("missing deletions");std::vector<uint64_t>deleted;uint64_t id;while(f>>id)deleted.push_back(id);
    Index idx(old);auto begin=Clock::now();for(auto p:points)idx.insert(p);double build=seconds(begin);
    idx.verify(queries);auto before=measure(idx,queries,repeats);
    begin=Clock::now();for(auto i:deleted)idx.erase(i);for(auto p:added)idx.insert(p);double update=seconds(begin);
    idx.verify(queries);auto shifted=measure(idx,queries,repeats);
    begin=Clock::now();size_t rekeyed=idx.rekey(next,regions);double rekey=seconds(begin);
    idx.verify(queries);auto after=measure(idx,queries,repeats);
    std::cout<<std::setprecision(12)<<"{\n"
      <<"\"build_seconds\":"<<build<<",\n"
      <<"\"query_before_us\":"<<before.latency<<",\n\"qps_before\":"<<before.qps<<",\n"
      <<"\"query_shifted_us\":"<<shifted.latency<<",\n\"qps_shifted\":"<<shifted.qps<<",\n"
      <<"\"query_adapted_us\":"<<after.latency<<",\n\"qps_adapted\":"<<after.qps<<",\n"
      <<"\"raw_update_seconds\":"<<update<<",\n\"reindex_seconds\":"<<rekey<<",\n"
      <<"\"update_operations\":"<<deleted.size()+added.size()<<",\n\"rekeyed_points\":"<<rekeyed<<",\n"
      <<"\"degradation_shifted_pct\":"<<100*(shifted.latency/before.latency-1)<<",\n"
      <<"\"degradation_adapted_pct\":"<<100*(after.latency/before.latency-1)<<",\n"
      <<"\"exact_results_verified\":true,\n\"checksums\":["<<before.checksum<<","<<shifted.checksum<<","<<after.checksum<<"]\n}\n";
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
