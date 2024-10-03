#include "storage/index/ivfflat_index.h"
#include <algorithm>
#include <optional>
#include <random>
#include "common/exception.h"
#include "execution/expressions/vector_expression.h"
#include "storage/index/index.h"
#include "storage/index/vector_index.h"

namespace bustub {
using Vector = std::vector<double>;

std::vector<Vector> IVFFlatIndex::RandomSample(const std::vector<std::pair<Vector, RID>> &data, size_t num_samples) {
    std::vector<Vector> sampled_centroids;

    // Check if we have enough data to sample from
    if (data.size() < num_samples) {
        throw std::invalid_argument("Not enough data to sample the required number of centroids");
    }

    // Create a vector of indices to shuffle
    std::vector<size_t> indices(data.size());
    for (size_t i = 0; i < data.size(); ++i) {
        indices[i] = i;
    }

    // Shuffle the indices using a random number generator
    std::random_device rd;
    std::mt19937 rng(rd());  // Random number generator
    std::shuffle(indices.begin(), indices.end(), rng);

    // Select the first 'num_samples' vectors based on the shuffled indices
    for (size_t i = 0; i < num_samples; ++i) {
        sampled_centroids.push_back(data[indices[i]].first);  // Add the vector to the sampled centroids
    }

    return sampled_centroids;
}

IVFFlatIndex::IVFFlatIndex(std::unique_ptr<IndexMetadata> &&metadata, BufferPoolManager *buffer_pool_manager,
                           VectorExpressionType distance_fn, const std::vector<std::pair<std::string, int>> &options)
    : VectorIndex(std::move(metadata), distance_fn) {
  std::optional<size_t> lists;
  std::optional<size_t> probe_lists;
  for (const auto &[k, v] : options) {
    if (k == "lists") {
      lists = v;
    } else if (k == "probe_lists") {
      probe_lists = v;
    }
  }
  if (!lists.has_value() || !probe_lists.has_value()) {
    throw Exception("missing options: lists / probe_lists for ivfflat index");
  }
  lists_ = *lists;
  probe_lists_ = *probe_lists;
}

void VectorAdd(Vector &a, const Vector &b) {
  for (size_t i = 0; i < a.size(); i++) {
    a[i] += b[i];
  }
}

void VectorScalarDiv(Vector &a, double x) {
  for (auto &y : a) {
    y /= x;
  }
}

// Find the nearest centroid to the base vector in all centroids
auto FindCentroid(const Vector &vec, const std::vector<Vector> &centroids, VectorExpressionType dist_fn) -> size_t {
  int min_index = -1;
  int min_distance = 0;
  for(int i = 0; i < centroids.size(); i++) {
    if(min_distance < ComputeDistance(vec, centroids[i], dist_fn)) {
      min_index = i;
      min_distance = ComputeDistance(vec, centroids[i], dist_fn);
    }
  }
  return min_index;
}

// Compute new centroids based on the original centroids.
auto FindCentroids(const std::vector<std::pair<Vector, RID>> &data, const std::vector<Vector> &centroids,
                   VectorExpressionType dist_fn) -> std::vector<Vector> {
  std::vector<Vector> res = centroids;
  std::vector<size_t> count(res.size(), 0);
  for(auto &it : data) {
    size_t index = FindCentroid(it.first, res, dist_fn);   
    VectorAdd(res[index],it.first);
    count[index]++;
  }
  for(int i = 0; i < res.size(); i++) {
    if(i == 0) continue;
    VectorScalarDiv(res[i], count[i]);  
  }
  return res;
}

void IVFFlatIndex::BuildIndex(std::vector<std::pair<Vector, RID>> initial_data) {
  if (initial_data.size() < lists_) {
    return;
  }
  centroids_buckets_.resize(lists_);
  centroids_ = RandomSample(initial_data, lists_);

  const size_t max_iterations = 500;
  for (size_t iter = 0; iter < max_iterations; ++iter) {
      // Update centroids using FindCentroids (which we've implemented earlier)
      centroids_ = FindCentroids(initial_data, centroids_, VectorExpressionType::L2Dist);
  }

  for (const auto& pair : initial_data) {
        const Vector& vec = pair.first;
        RID rid = pair.second;

        // 找到最近的中心点
        size_t nearest_centroid_idx = FindCentroid(vec, centroids_, VectorExpressionType::L2Dist);

        // 将 RID 添加到对应的倒排索引列表中
        centroids_buckets_[nearest_centroid_idx].push_back(pair);
  }
  return ;
  
  
  // IMPLEMENT ME
}

void IVFFlatIndex::InsertVectorEntry(const std::vector<double> &key, RID rid) {
  size_t nearest_centroid_idx = FindCentroid(key, centroids_, VectorExpressionType::L2Dist);
  centroids_buckets_[nearest_centroid_idx].emplace_back(key,rid);
} 

std::vector<size_t> IVFFlatIndex::FindNearestCentroids(const std::vector<double> &base_vector, size_t num_centroids) {
    // 存储 (距离, 中心点索引) 的 pair
    std::vector<std::pair<double, size_t>> distances;

    // 遍历所有的中心点，计算每个中心点与 base_vector 的距离
    for (size_t i = 0; i < centroids_.size(); ++i) {
        // 计算 base_vector 和当前中心点的 L2 距离
        double dist = ComputeDistance(base_vector, centroids_[i], VectorExpressionType::L2Dist);

        // 将距离和索引存入 distances 向量中
        distances.emplace_back(dist, i);
    }

    // 按距离升序排序
    std::sort(distances.begin(), distances.end(), 
              [](const std::pair<double, size_t>& a, const std::pair<double, size_t>& b) {
                  return a.first < b.first;  // 比较距离，按升序排序
              });

    // 选择距离最近的 num_centroids 个中心点的索引
    std::vector<size_t> nearest_centroids;
    for (size_t i = 0; i < num_centroids && i < distances.size(); ++i) {
        nearest_centroids.push_back(distances[i].second);  // 只存储中心点的索引
    }

    return nearest_centroids;  // 返回最近的中心点的索引列表
}


auto IVFFlatIndex::ScanVectorKey(const std::vector<double> &base_vector, size_t limit) -> std::vector<RID> {
    std::vector<RID> global_result;  // 全局结果，存储最终的最近邻居
    std::vector<std::pair<double, RID>> local_results;  // 存储局部结果 (distance, RID)

    // Step 1: 找到与 base_vector 距离最近的 probe_lists_ 个中心点
    std::vector<size_t> nearest_centroids = FindNearestCentroids(base_vector, probe_lists_);

    // Step 2: 遍历每个中心点的倒排列表，获取局部结果
    for (size_t centroid_idx : nearest_centroids) {
        for (const auto& entry : centroids_buckets_[centroid_idx]) {
            const Vector& vec = entry.first;  // 获取存储的向量
            RID rid = entry.second;  // 获取对应的 RID

            // 计算当前向量与 base_vector 的距离
            double distance = ComputeDistance(base_vector, vec, VectorExpressionType::L2Dist);

            // 将 (distance, RID) 插入局部结果中
            local_results.emplace_back(distance, rid);
        }
    }

    // Step 3: 对局部结果进行排序，取前 limit 个最近邻居
    std::sort(local_results.begin(), local_results.end(), 
              [](const std::pair<double, RID>& a, const std::pair<double, RID>& b) {
                  return a.first < b.first;  // 根据距离升序排序
              });

    // Step 4: 取前 limit 个 RID 作为最终结果
    for (size_t i = 0; i < std::min(limit, local_results.size()); ++i) {
        global_result.push_back(local_results[i].second);
    }

    return global_result;  // 返回最终的最近邻居的 RID 列表
}



}  // namespace bustub
