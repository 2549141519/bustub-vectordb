#include "execution/executors/sort_executor.h"

namespace bustub {

SortExecutor::SortExecutor(ExecutorContext *exec_ctx, const SortPlanNode *plan,
                           std::unique_ptr<AbstractExecutor> &&child_executor)
    : AbstractExecutor(exec_ctx),plan_(plan),child_executor_(std::move(child_executor)) {}
    //sort_node存储这vector 里边存储具体的expression 由child_excutor读出来
    //我们还需要要目标向量 距离类型 排序规则 这些存储在vector的第一个数据里 
void SortExecutor::Init() {
    child_executor_->Init();
    if(child_executor_ == nullptr)  return;
    Tuple tuple_from_child;
    RID rid_from_child;
    
    while (child_executor_->Next(&tuple_from_child, &rid_from_child)) {
    // 将元组保存到列表中
        tuples.push_back(std::make_pair(rid_from_child,tuple_from_child));
    }
    auto schema = child_executor_->GetOutputSchema();
    // Step 3: 使用 order_bys_ 中的表达式对元组进行排序
   std::sort(tuples.begin(), tuples.end(),
          [&](const std::pair<RID, Tuple> &a, const std::pair<RID, Tuple> &b) {
              // 提取 pair 中的 Tuple
              const Tuple &tuple_a = a.second;
              const Tuple &tuple_b = b.second;

              // 遍历所有的 order_bys_ 来确定排序顺序
              for (const auto &[order_type, expr] : plan_->GetOrderBy()) {
                  // Evaluate the expressions on both tuples
                  Value val_a = expr->Evaluate(&tuple_a, schema);
                  Value val_b = expr->Evaluate(&tuple_b, schema);

                  // 比较两个元组的表达式值
                  if (val_a.GetAs<double>() < val_b.GetAs<double>()) {
                      return order_type == OrderByType::DEFAULT || order_type == OrderByType::ASC;
                  } else if (val_a.GetAs<double>() > val_b.GetAs<double>()) {
                      return order_type == OrderByType::DESC;
                  }
              }

              // 如果所有字段相等，返回 false
              return false;
          });

          iter_ = tuples.begin();
}

auto SortExecutor::Next(Tuple *tuple, RID *rid) -> bool { 
    if(iter_ != tuples.end()) {
        *tuple = iter_->second;
        *rid = iter_->first;
        iter_++;
        return true;
    }
    return false;
 }

}  // namespace bustub
