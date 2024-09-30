#include "optimizer/optimizer.h"


namespace bustub {

auto Optimizer::OptimizeSortLimitAsTopN(const AbstractPlanNodeRef &plan) -> AbstractPlanNodeRef {
  // TODO(student): implement sort + limit -> top N optimizer rule
  //TopNPlanNode(SchemaRef output, AbstractPlanNodeRef child,
  //             std::vector<std::pair<OrderByType, AbstractExpressionRef>> order_bys, std::size_t n)
  SchemaRef schema_ref = std::make_shared<const Schema>(plan->OutputSchema());
  auto limit_plan = std::dynamic_pointer_cast<const LimitPlanNode>(plan);
  if (!limit_plan) {
    return plan; // 如果转换失败，返回原计划
  }
  auto sort_plan = std::dynamic_pointer_cast<const SortPlanNode>(plan->children_[0]);
  if (!sort_plan) {
    return plan; // 如果转换失败，返回原计划
  }
  auto seq_plan = std::dynamic_pointer_cast<const SeqScanPlanNode>(plan->children_[0]);
  if (!seq_plan) {
    return plan; // 如果转换失败，返回原计划
  }
  auto res = std::make_shared<TopNPlanNode>(schema_ref,seq_plan, 
    sort_plan->GetOrderBy(),limit_plan->limit_);
  return res;
}

}  // namespace bustub
