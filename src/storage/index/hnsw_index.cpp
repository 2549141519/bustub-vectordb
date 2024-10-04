#include <algorithm>
#include <cmath>
#include <functional>
#include <iterator>
#include <memory>
#include <queue>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "common/macros.h"
#include "execution/expressions/vector_expression.h"
#include "fmt/format.h"
#include "fmt/std.h"
#include "storage/index/hnsw_index.h"
#include "storage/index/index.h"
#include "storage/index/vector_index.h"

namespace bustub {
struct CompareByDistance {
  bool operator()(const std::pair<double, size_t> &a, const std::pair<double, size_t> &b) const {
    return a.first < b.first;  // 这样将保持最大堆，距离大的在堆顶
  }
};
HNSWIndex::HNSWIndex(std::unique_ptr<IndexMetadata> &&metadata, BufferPoolManager *buffer_pool_manager,
                     VectorExpressionType distance_fn, const std::vector<std::pair<std::string, int>> &options)
    : VectorIndex(std::move(metadata), distance_fn),
      vertices_(std::make_unique<std::vector<Vector>>()),
      layers_{{*vertices_, distance_fn}} {
  std::optional<size_t> m;
  std::optional<size_t> ef_construction;
  std::optional<size_t> ef_search;
  for (const auto &[k, v] : options) {
    if (k == "m") {
      m = v;
    } else if (k == "ef_construction") {
      ef_construction = v;
    } else if (k == "ef_search") {
      ef_search = v;
    }
  }
  if (!m.has_value() || !ef_construction.has_value() || !ef_search.has_value()) {
    throw Exception("missing options: m / ef_construction / ef_search for hnsw index");
  }
  ef_construction_ = *ef_construction;
  m_ = *m;
  ef_search_ = *ef_search;
  m_max_ = m_;
  m_max_0_ = m_ * m_;
  layers_[0].m_max_ = m_max_0_;
  m_l_ = 1.0 / std::log(m_);
  std::random_device rand_dev;
  generator_ = std::mt19937(rand_dev());
}

auto SelectNeighbors(const std::vector<double> &vec, const std::vector<size_t> &vertex_ids,
                     const std::vector<std::vector<double>> &vertices, size_t m, VectorExpressionType dist_fn)
    -> std::vector<size_t> {
// 定义比较器，根据 ComputeDistance 计算的距离来进行排序
  auto cmp = [&](size_t left_id, size_t right_id) {
  double dist_left = ComputeDistance(vec, vertices[left_id], dist_fn);
  double dist_right = ComputeDistance(vec, vertices[right_id], dist_fn);
  return dist_left > dist_right;  // 按距离升序排列
};

  // 使用优先队列（堆）来维护最近的 m 个邻居，堆顶是距离最大的邻居
  std::priority_queue<size_t, std::vector<size_t>, decltype(cmp)> min_heap(cmp);

  // 遍历候选顶点ID
  for (const auto &vertex_id : vertex_ids) {
    // 将顶点ID插入堆中
    min_heap.push(vertex_id);

    // 如果堆的大小超过了 m，则弹出堆顶元素（最远的元素）
    if (min_heap.size() > m) {
      min_heap.pop();
    }
  }

  // 提取最近的 m 个顶点ID
  std::vector<size_t> nearest_neighbors;
  while (!min_heap.empty()) {
    nearest_neighbors.push_back(min_heap.top());  // 获取顶点ID
    min_heap.pop();  // 移除堆顶
  }

  // 返回按距离最近的 m 个顶点ID
  return nearest_neighbors;
}

auto NSW::SearchLayer(const std::vector<double> &base_vector, size_t limit, const std::vector<size_t> &entry_points)
    -> std::vector<size_t> {
  // 使用 std::queue 作为访问者队列
  std::queue<size_t> candidate_queue;

  // 结果集合，最大堆，用于保存最近的 limit 个邻居
  std::priority_queue<std::pair<double, size_t>, std::vector<std::pair<double, size_t>>, CompareByDistance> result_set;

  // 记录已经访问过的节点，防止重复访问
  std::unordered_set<size_t> visited;

  // 当前结果集中的最远距离
  double max_result_dist = std::numeric_limits<double>::min();

  // 当前候选队列中最小的距离
  double min_candidate_dist = std::numeric_limits<double>::max();

  // 将所有入口点加入访问者队列和结果集合
  for (const auto &entry_point : entry_points) {
    double dist = ComputeDistance(base_vector, vertices_[entry_point], dist_fn_);
    candidate_queue.push(entry_point);
    result_set.emplace(dist, entry_point);
    visited.insert(entry_point);

    // 更新结果集中最远距离
    if (result_set.size() == limit) {
      max_result_dist = result_set.top().first;
    }

    // 更新候选队列的最小距离
    min_candidate_dist = std::min(min_candidate_dist, dist);
  }

  // 开始搜索
  while (!candidate_queue.empty()) {
    size_t curr_vertex = candidate_queue.front();
    candidate_queue.pop();

    const auto &neighbors = edges_[curr_vertex];
    auto nearest_neighbors = SelectNeighbors(base_vector, neighbors, vertices_, limit, dist_fn_);

    for (const auto &neighbor : nearest_neighbors) {
      if (visited.find(neighbor) != visited.end()) continue;

      double neighbor_dist = ComputeDistance(base_vector, vertices_[neighbor], dist_fn_);
      visited.insert(neighbor);
      candidate_queue.push(neighbor);

      result_set.emplace(neighbor_dist, neighbor);

      // 如果结果集超过了限制大小，移除最远的节点
      if (result_set.size() > limit) {
        result_set.pop();
      }

      // 更新结果集中最远的距离
      if (result_set.size() == limit) {
        max_result_dist = result_set.top().first;
      }

      // 更新候选队列的最小距离
      min_candidate_dist = std::min(min_candidate_dist, neighbor_dist);
    }

    // 检查终止条件
    if (result_set.size() == limit && min_candidate_dist > max_result_dist) {
      // 候选队列中的最小距离已经大于或等于结果集中的最远距离，停止搜索
      break;
    }
  }

  // 提取结果集合中的最近邻居
  std::vector<size_t> nearest_neighbors;
  while (!result_set.empty()) {
    nearest_neighbors.push_back(result_set.top().second);
    result_set.pop();
  }
  
  std::reverse(nearest_neighbors.begin(), nearest_neighbors.end());
  return nearest_neighbors;
}

auto NSW::AddVertex(size_t vertex_id) { in_vertices_.push_back(vertex_id); }

auto NSW::Insert(const std::vector<double> &vec, size_t vertex_id, size_t ef_construction, size_t m) {
  // IMPLEMENT ME
  AddVertex(vertex_id);
  
  // Step 2: 找到与新向量最接近的 m 个邻居
  auto nearest_neighbors = SearchLayer(vec, m, {DefaultEntryPoint()});
  
  // Step 3: 将新顶点与最近邻居连接
  for (const auto &neighbor_id : nearest_neighbors) {
    Connect(vertex_id, neighbor_id);  // 双向连接新顶点和邻居
  }

  // Step 4: 调整连接数，确保每个节点的连接数不超过 max_m
  for (const auto &neighbor_id : nearest_neighbors) {
    if (edges_[neighbor_id].size() > m_max_) {
      // 如果邻居的连接数超过 max_m，重新计算最近邻居并调整连接数
      auto neighbors_to_keep = SelectNeighbors(vertices_[neighbor_id], edges_[neighbor_id], vertices_, m_max_, dist_fn_);
      edges_[neighbor_id] = neighbors_to_keep;  // 只保留最近的 max_m 个邻居
    }
  }
}

void NSW::Connect(size_t vertex_a, size_t vertex_b) {
  edges_[vertex_a].push_back(vertex_b);
  edges_[vertex_b].push_back(vertex_a);
}

auto HNSWIndex::AddVertex(const std::vector<double> &vec, RID rid) -> size_t {
  auto id = vertices_->size();
  vertices_->emplace_back(vec);
  rids_.emplace_back(rid);
  return id;
}

void HNSWIndex::BuildIndex(std::vector<std::pair<std::vector<double>, RID>> initial_data) {
  std::shuffle(initial_data.begin(), initial_data.end(), generator_);

  for (const auto &[vec, rid] : initial_data) {
    InsertVectorEntry(vec, rid);
  }
}
int HNSWIndex::GenerateRandomLevel() {
    // 生成 [0, 1] 范围内均匀分布的随机数
    std::uniform_real_distribution<> uniform_dist(0.0, 1.0);
    double random_value = uniform_dist(generator_);  // 使用 generator_ 生成随机数

    // 根据公式计算插入的层级
    int level = static_cast<int>(-std::log(random_value) * m_l_);

    return level;  // 返回生成的层级
}


auto HNSWIndex::ScanVectorKey(const std::vector<double> &base_vector, size_t limit) -> std::vector<RID> {
  std::vector<size_t> entry_points = {layers_.rbegin()->DefaultEntryPoint()};
  int i = layers_.size() - 1;
  while(i > 0) {
    entry_points = layers_[i--].SearchLayer(base_vector, limit, entry_points);
  }
  entry_points = layers_[0].SearchLayer(base_vector, limit, entry_points);
  std::vector<RID> result;
  result.reserve(entry_points.size());
  for (const auto &id : entry_points) {
    result.push_back(rids_[id]);
  }
  return result;
}

void HNSWIndex::InsertVectorEntry(const std::vector<double> &key, RID rid) {
  std::uniform_real_distribution<double> level_dist(0.0, 1.0);
  auto vertex_id = AddVertex(key, rid);
  int target_level = static_cast<int>(std::floor(-std::log(level_dist(generator_)) * m_l_));
  
  std::vector<size_t> nearest_elements;
  if (!layers_[0].in_vertices_.empty()) {
    std::vector<size_t> entry_points{layers_[layers_.size() - 1].DefaultEntryPoint()};
    int level = layers_.size() - 1;
    for (; level > target_level; level--) {
      nearest_elements = layers_[level].SearchLayer(key, ef_search_, entry_points);
      nearest_elements = SelectNeighbors(key, nearest_elements, *vertices_, 1, distance_fn_);
      entry_points = {nearest_elements[0]};
    }
    for (; level >= 0; level--) {
      auto &layer = layers_[level];
      nearest_elements = layer.SearchLayer(key, ef_construction_, entry_points);
      auto neighbors = SelectNeighbors(key, nearest_elements, *vertices_, m_, distance_fn_);
      layer.AddVertex(vertex_id);
      for (const auto neighbor : neighbors) {
        layer.Connect(vertex_id, neighbor);
      }
      for (const auto neighbor : neighbors) {
        auto &edges = layer.edges_[neighbor];
        if (edges.size() > m_max_) {
          auto new_neighbors = SelectNeighbors((*vertices_)[neighbor], edges, *vertices_, layer.m_max_, distance_fn_);
          edges = new_neighbors;
        }
      }
      entry_points = nearest_elements;
    }
  } else {
    layers_[0].AddVertex(vertex_id);
  }
  while (static_cast<int>(layers_.size()) <= target_level) {
    auto layer = NSW{*vertices_, distance_fn_, m_max_};
    layer.AddVertex(vertex_id);
    layers_.emplace_back(std::move(layer));
  }

  
}

}  // namespace bustub
