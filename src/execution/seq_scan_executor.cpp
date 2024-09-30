//===----------------------------------------------------------------------===//
//
//                         BusTub
//
// seq_scan_executor.cpp
//
// Identification: src/execution/seq_scan_executor.cpp
//
// Copyright (c) 2015-2021, Carnegie Mellon University Database Group
//
//===----------------------------------------------------------------------===//

#include "execution/executors/seq_scan_executor.h"

namespace bustub {

SeqScanExecutor::SeqScanExecutor(ExecutorContext *exec_ctx, const SeqScanPlanNode *plan) : AbstractExecutor(exec_ctx),plan_(plan) {}

void SeqScanExecutor::Init() {
    // 获取表的 OID（从计划节点中）
  table_oid_t table_oid = plan_->GetTableOid();

  // 从执行上下文中获取目录（catalog），并找到对应的表信息
  TableInfo *table_info = exec_ctx_->GetCatalog()->GetTable(table_oid);

  // 使用 table_info 来获取表堆
  table_heap_ = table_info->table_.get();

  // 创建表的迭代器，指向表的第一个元组
  iter_ = std::make_unique<TableIterator>(table_heap_->MakeIterator());

  // 表的迭代器已初始化，可以通过 iter_ 进行顺序扫描
}

auto SeqScanExecutor::Next(Tuple *tuple, RID *rid) -> bool {
     // 如果迭代器已经到达表的末尾，返回 false
    if (iter_->IsEnd()) {
        return false;
    }

    // 获取当前元组及其 RID
    *tuple = iter_->GetTuple().second;
    *rid = tuple->GetRid();

    // 将迭代器移动到下一个元组
    ++(*iter_);

    // 返回 true，表示成功获取了元组
    return true;
}  
}   // namespace bustub
