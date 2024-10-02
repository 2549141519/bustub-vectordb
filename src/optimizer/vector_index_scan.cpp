#include <memory>
#include <optional>
#include "binder/bound_order_by.h"
#include "catalog/catalog.h"
#include "catalog/column.h"
#include "concurrency/transaction.h"
#include "execution/expressions/array_expression.h"
#include "execution/expressions/column_value_expression.h"
#include "execution/expressions/comparison_expression.h"
#include "execution/expressions/constant_value_expression.h"
#include "execution/expressions/vector_expression.h"
#include "execution/plans/abstract_plan.h"
#include "execution/plans/index_scan_plan.h"
#include "execution/plans/limit_plan.h"
#include "execution/plans/projection_plan.h"
#include "execution/plans/seq_scan_plan.h"
#include "execution/plans/sort_plan.h"
#include "execution/plans/vector_index_scan_plan.h"
#include "fmt/core.h"
#include "optimizer/optimizer.h"
#include "type/type.h"
#include "type/type_id.h"

namespace bustub {

auto MatchVectorIndex(const Catalog &catalog, table_oid_t table_oid, uint32_t col_idx, VectorExpressionType dist_fn,
                      std::string &vector_index_match_method) -> const IndexInfo * {
  // 从 Catalog 中获取表的所有索引，通过表的 OID 获取表名
  auto table_info = catalog.GetTable(table_oid);
  if (table_info == nullptr) {
    return nullptr; // 如果表不存在，返回 nullptr
  }

  // 获取该表的所有索引
  auto table_indexes = catalog.GetTableIndexes(table_info->name_);

  // 遍历表的所有索引
  for (const auto &index_info : table_indexes) {
    // 检查索引的列是否匹配指定的向量列索引 (col_idx)
    if (index_info->key_schema_.GetColumns().size() > col_idx) {
      // 如果索引的列数足够，并且指定列匹配，继续检查其他属性
      const auto &column = index_info->key_schema_.GetColumn(col_idx);

      // 检查索引的类型是否为向量索引
      if (index_info->index_type_ == IndexType::VectorHNSWIndex || index_info->index_type_ == IndexType::VectorIVFFlatIndex) {
        // 检查距离函数是否匹配
        bool dist_fn_match = false;
        if ((dist_fn == VectorExpressionType::L2Dist && column.GetType() == TypeId::VECTOR) ||
            (dist_fn == VectorExpressionType::CosineSimilarity && column.GetType() == TypeId::VECTOR) ||
            (dist_fn == VectorExpressionType::InnerProduct && column.GetType() == TypeId::VECTOR)) {
          dist_fn_match = true;
        }

        if (dist_fn_match) {
          // 根据 vector_index_match_method 选择匹配策略
          if (vector_index_match_method.empty() || vector_index_match_method == "default") {
            // 选择第一个匹配的向量索引
            vector_index_match_method = index_info->name_;
            return index_info;
          } else if (vector_index_match_method == "hnsw" && index_info->index_type_ == IndexType::VectorHNSWIndex) {
            vector_index_match_method = index_info->name_;
            return index_info;
          } else if (vector_index_match_method == "ivfflat" && index_info->index_type_ == IndexType::VectorIVFFlatIndex) {
            vector_index_match_method = index_info->name_;
            return index_info;
          } else if (vector_index_match_method == "none") {
            // 如果索引匹配策略为 "none"，不返回任何索引，执行顺序扫描
            return nullptr;
          }
        }
      }
    }
  }

  // 如果没有找到合适的索引，返回 nullptr
  return nullptr;
}



auto Optimizer::OptimizeAsVectorIndexScan(const AbstractPlanNodeRef &plan) -> AbstractPlanNodeRef {
  std::vector<AbstractPlanNodeRef> children;

  // 递归优化子计划
  for (const auto &child : plan->GetChildren()) {
    children.emplace_back(OptimizeAsVectorIndexScan(child));
  }
  if(plan->GetType() == PlanType::SeqScan || plan->GetType() == PlanType::Projection)
    return plan;
  // 如果当前计划是 SeqScan 或者 Projection
  std::shared_ptr<const TopNPlanNode> topn_node = nullptr;
  std::shared_ptr<const ProjectionPlanNode> projection_plan = nullptr;
  std::shared_ptr<const SeqScanPlanNode> seq_scan_plan = nullptr;
  VectorExpressionType dist_fn = VectorExpressionType::L2Dist;  // 默认距离函数类型
  
  
  topn_node = std::dynamic_pointer_cast<const TopNPlanNode>(plan);
  size_t limit = 0;
  std::shared_ptr<const ArrayExpression> base_vector = nullptr;
  if(topn_node != nullptr) {
    limit = topn_node->GetN();
    std::vector<AbstractExpressionRef> array_children;
    array_children.push_back(topn_node->GetOrderBy()[0].second); // 假设你从 OrderBy 中获取了正确的表达式
    base_vector = std::make_shared<ArrayExpression>(array_children);
  }
  else
    return plan;
  

  // 如果是 Projection，则从其子计划中找到 SeqScan
  if (plan->children_[0]->GetType() == PlanType::Projection) {
    projection_plan = std::dynamic_pointer_cast<const ProjectionPlanNode>(plan->children_[0]);
    seq_scan_plan = std::dynamic_pointer_cast<const SeqScanPlanNode>(projection_plan->GetChildAt(0));
  } else if (plan->children_[0]->GetType() == PlanType::SeqScan) {
    seq_scan_plan = std::dynamic_pointer_cast<const SeqScanPlanNode>(plan->children_[0]);
  }

  // 检查是否找到 SeqScan 计划
  if (seq_scan_plan != nullptr) {
    // 假设要查找的向量列是第 0 列
    uint32_t col_idx = 0;
    
    // 如果是 Projection，找到投影中的距离函数
    if (projection_plan != nullptr) {
      for (const auto &expr : projection_plan->GetExpressions()) {
        if (auto vector_expr = std::dynamic_pointer_cast<VectorExpression>(expr)) {
          dist_fn = vector_expr->expr_type_;  // 提取距离计算函数类型        
        }
      }
    }

    // 从 SeqScan 计划中提取列和距离函数（假设第 0 列是向量列）
    std::string vector_index_match_method = !vector_index_match_method_.empty() ? vector_index_match_method_ : "default";
    const IndexInfo *index_info = MatchVectorIndex(catalog_, seq_scan_plan->GetTableOid(), col_idx, dist_fn, vector_index_match_method);

    // 如果找到可用的向量索引，替换顺序扫描为向量索引扫描
    if (index_info != nullptr) {
      /*
       VectorIndexScanPlanNode(SchemaRef output, table_oid_t table_oid,
        std::string table_name, index_oid_t index_oid,
        std::string index_name, std::shared_ptr<const ArrayExpression> base_vector, size_t limit)
      */
      auto sch = std::make_shared<Schema>(seq_scan_plan->OutputSchema());
      std::shared_ptr<AbstractPlanNode> vector_index_scan_plan = std::make_shared<VectorIndexScanPlanNode>(
          sch,      // 输出 schema
          seq_scan_plan->GetTableOid(),
          catalog_.GetTable(seq_scan_plan->GetTableOid())->name_,
          index_info->index_oid_, 
          vector_index_match_method,
          base_vector,
          limit                                     
      );

      // 如果原计划中有 Projection，则需要在 VectorIndexScan 之后添加一个投影
      if (projection_plan != nullptr) {
        auto schema_ref = std::make_shared<const Schema>(projection_plan->OutputSchema());
        std::vector<AbstractExpressionRef> expressions = projection_plan->GetExpressions();
        std::shared_ptr<AbstractPlanNode> res = std::make_shared<ProjectionPlanNode>(schema_ref, expressions,
        nullptr);
        res->children_.push_back(vector_index_scan_plan);
        return res;
      }
      return vector_index_scan_plan;
    }
  }

  // 如果无法优化或不是 SeqScan 或 Projection 计划，返回优化后的计划
  return plan;
}


}  // namespace bustub
