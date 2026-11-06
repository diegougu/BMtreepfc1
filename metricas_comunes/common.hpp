#pragma once
#include <bits/stdc++.h>
using Point = std::array<double, 2>;
using Clock = std::chrono::steady_clock;
struct Box {
    Point lo, hi;
    bool contains(const Point& p) const {
        return lo[0] <= p[0] && p[0] <= hi[0] && lo[1] <= p[1] && p[1] <= hi[1];
    }
};
inline double elapsed(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}
inline std::vector<Point> points_file(const std::string& path) {
    std::ifstream f(path);
    if (!f) { throw std::runtime_error("No se puede leer " + path); }
    std::vector<Point> result;
    Point p;
    while (f >> p[0] >> p[1]) { result.push_back(p); }
    if (!f.eof()) { throw std::runtime_error("Puntos invalidos"); }
    return result;
}
inline std::vector<Box> boxes_file(const std::string& path) {
    std::ifstream f(path);
    if (!f) { throw std::runtime_error("No se puede leer " + path); }
    std::vector<Box> result;
    Box b;
    while (f >> b.lo[0] >> b.lo[1] >> b.hi[0] >> b.hi[1]) {
        if (b.lo[0] > b.hi[0] || b.lo[1] > b.hi[1]) { throw std::runtime_error("Limites invalidos"); }
        result.push_back(b);
    }
    if (!f.eof()) { throw std::runtime_error("Consulta invalida"); }
    return result;
}
struct Workload {
    std::vector<Point> initial, additions, final_points;
    std::vector<size_t> deletions;
    std::vector<Box> training, test, regions;
    explicit Workload(const std::string& dir) {
        initial = points_file(dir + "/points.tsv");
        additions = points_file(dir + "/insertions.tsv");
        training = boxes_file(dir + "/train_queries.tsv");
        test = boxes_file(dir + "/test_queries.tsv");
        regions = boxes_file(dir + "/regions.tsv");
        if (initial.empty() || training.empty() || test.empty()) { throw std::runtime_error("Carga vacia"); }
        std::ifstream f(dir + "/deletions.tsv");
        if (!f) { throw std::runtime_error("Faltan eliminaciones"); }
        size_t id;
        std::set<size_t> removed;
        while (f >> id) {
            if (id >= initial.size() || !removed.insert(id).second) { throw std::runtime_error("Eliminacion invalida"); }
            deletions.push_back(id);
        }
        std::set<Point> unique(initial.begin(), initial.end());
        if (unique.size() != initial.size()) { throw std::runtime_error("La comparacion requiere puntos unicos"); }
        for (auto i : deletions) { unique.erase(initial[i]); }
        for (const auto& p : additions) {
            if (!unique.insert(p).second) { throw std::runtime_error("Insercion duplicada"); }
        }
        final_points.assign(unique.begin(), unique.end());
    }
};
struct Validation {
    uint64_t true_positives = 0, expected = 0, false_positives = 0, mismatched = 0;
    double recall() const { return expected ? double(true_positives)/expected : 1.0; }
};
template<class Index>
Validation validate(Index& index, const std::vector<Point>& points, const std::vector<Box>& queries) {
    Validation v;
    for (const auto& q : queries) {
        auto actual = index.enumerate(q);
        std::vector<Point> expected, intersection;
        for (const auto& p : points) { if (q.contains(p)) { expected.push_back(p); } }
        std::sort(actual.begin(), actual.end());
        std::sort(expected.begin(), expected.end());
        std::set_intersection(actual.begin(), actual.end(), expected.begin(), expected.end(), std::back_inserter(intersection));
        v.true_positives += intersection.size();
        v.expected += expected.size();
        v.false_positives += actual.size() - intersection.size();
        v.mismatched += actual != expected;
    }
    return v;
}
struct QueryMetrics { double us, qps; uint64_t checksum; };
template<class Index>
QueryMetrics measure(Index& index, const std::vector<Box>& queries, int repeats) {
    uint64_t checksum = 0;
    for (const auto& q : queries) { checksum += index.enumerate(q).size(); }
    auto start = Clock::now();
    for (int r = 0; r < repeats; ++r) {
        for (const auto& q : queries) { checksum += index.enumerate(q).size(); }
    }
    double seconds = elapsed(start);
    double n = double(queries.size()) * repeats;
    return {seconds * 1e6 / n, n / seconds, checksum};
}
inline void emit(const QueryMetrics& before, const QueryMetrics& after, const Validation& vb,
                 const Validation& va, double build, double updates, double adaptation,
                 size_t operations, size_t memory_before, size_t memory_after,
                 const std::string& extra = "") {
    std::cout << std::setprecision(15)
        << "{\"query_before_us\":" << before.us
        << ",\"query_after_us\":" << after.us
        << ",\"qps\":" << after.qps
        << ",\"qps_before\":" << before.qps
        << ",\"degradation_pct\":" << 100*(after.us/before.us-1)
        << ",\"build_runtime_seconds\":" << build
        << ",\"update_runtime_seconds\":" << updates
        << ",\"adaptation_runtime_seconds\":" << adaptation
        << ",\"update_operations\":" << operations
        << ",\"index_bytes_before\":" << memory_before
        << ",\"index_bytes_after\":" << memory_after
        << ",\"recall\":" << va.recall()
        << ",\"recall_before\":" << vb.recall()
        << ",\"false_positives\":" << va.false_positives
        << ",\"mismatched_queries\":" << va.mismatched
        << ",\"exact_results_verified\":" << ((va.mismatched==0 && vb.mismatched==0)?"true":"false")
        << ",\"checksum_before\":" << before.checksum
        << ",\"checksum_after\":" << after.checksum
        << extra << "}\n";
}
