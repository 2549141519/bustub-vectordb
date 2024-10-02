#include "execution/executors/topn_executor.h"

#include <queue>
#include <vector>
#include <functional> 
namespace bustub {

TopNExecutor::TopNExecutor(ExecutorContext *exec_ctx, const TopNPlanNode *plan,
                           std::unique_ptr<AbstractExecutor> &&child_executor)
    : AbstractExecutor(exec_ctx),plan_(plan),child_executor_(std::move(child_executor)){}

void TopNExecutor::Init() {
    child_executor_->Init();
    if (child_executor_ == nullptr) return;

    Tuple tuple_from_child;
    RID rid_from_child;
    auto schema = child_executor_->GetOutputSchema();

    // 定义一个优先队列（堆），根据排序类型选择不同的比较器
    auto cmp = [&](const std::pair<RID, Tuple> &a, const std::pair<RID, Tuple> &b) {
        const Tuple &tuple_a = a.second;
        const Tuple &tuple_b = b.second;

        for (const auto &[order_type, expr] : plan_->GetOrderBy()) {
            Value val_a = expr->Evaluate(&tuple_a, schema);
            Value val_b = expr->Evaluate(&tuple_b, schema);

            // 比较两个元组的表达式值
            if (val_a.GetAs<double>() < val_b.GetAs<double>()) {
                return order_type == OrderByType::DEFAULT || order_type == OrderByType::ASC;
            } else if (val_a.GetAs<double>() > val_b.GetAs<double>()) {
                return order_type == OrderByType::DESC;
            }
        }
        return false;  // 所有字段都相等时，返回 false
    };

    using PairType = std::pair<RID, Tuple>;
    std::priority_queue<PairType, std::vector<PairType>, decltype(cmp)> pq(cmp);

    size_t k = plan_->GetN();

     // 遍历元组，维护前 k 个元素
    while (child_executor_->Next(&tuple_from_child, &rid_from_child)) {
        pq.push(std::make_pair(rid_from_child, tuple_from_child));

        // 如果堆的大小超过 k 个元素，则删除一个元素
        if (pq.size() > k) {
            pq.pop();  // 移除堆顶元素（最大或最小的元素，具体取决于比较器）
        }
    }
    while (!pq.empty()) {
        tuples.push_back(pq.top());
        pq.pop();
    }
    
    // 迭代器初始化
    iter_ = tuples.rbegin();
}

auto TopNExecutor::Next(Tuple *tuple, RID *rid) -> bool {
    if(limit++ < plan_->GetN() && iter_ != tuples.rend()) {
        *tuple = iter_->second;
        *rid = iter_->first;
        iter_++;
        return true;
    }
    return false;
}

auto TopNExecutor::GetNumInHeap() -> size_t { throw NotImplementedException("TopNExecutor is not implemented"); };

}  // namespace bustub
