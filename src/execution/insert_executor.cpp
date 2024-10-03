//===----------------------------------------------------------------------===//
//
//                         BusTub
//
// insert_executor.cpp
//
// Identification: src/execution/insert_executor.cpp
//
// Copyright (c) 2015-2021, Carnegie Mellon University Database Group
//
//===----------------------------------------------------------------------===//

#include <memory>

#include "execution/executors/insert_executor.h"


namespace bustub {

InsertExecutor::InsertExecutor(ExecutorContext *exec_ctx, const InsertPlanNode *plan,
                               std::unique_ptr<AbstractExecutor> &&child_executor)
    : AbstractExecutor(exec_ctx),  // 初始化基类 AbstractExecutor
      plan_(plan),                 // 初始化 InsertPlanNode
      child_executor_(std::move(child_executor))  // 初始化子执行器
{   
    // 初始化表堆
    table_heap_ = exec_ctx->GetCatalog()->GetTable(plan->GetTableOid())->table_.get();
   
    // 初始化表的向量索引
    /////////////////////////
    
    
    // 初始化状态变量
    emitted_ = false;
}

void InsertExecutor::Init() { 
    // 假设 plan_ 是 AbstractPlanNode 类型
    child_executor_->Init();
 }


auto InsertExecutor::Next(Tuple *tuple, RID *rid) -> bool { 
    if (child_executor_ != nullptr) {
        if (child_executor_->Next(tuple, rid)) {
                Transaction *txn = exec_ctx_->GetTransaction();
                LockManager *lock_mgr = exec_ctx_->GetLockManager();
                auto table_oid = plan_->GetTableOid();
                TupleMeta tuple_meta;
                tuple_meta.is_deleted_ = false;  // 插入时元组不应标记为已删除
                // 调用 InsertTuple，传递所有必要的参数：元数据、元组、锁管理器、事务、表 OID
                auto inserted_rid = table_heap_->InsertTuple(tuple_meta, *tuple, lock_mgr, txn, table_oid);
                if (inserted_rid.has_value()) {
                    auto table_indexes = exec_ctx_->GetCatalog()->GetTableIndexes(
                        exec_ctx_->GetCatalog()->GetTable(plan_->GetTableOid())->name_);
                    *rid = inserted_rid.value();
                    for (auto &index_info : table_indexes) {
                        // 插入到索引中
                            auto vector_index = dynamic_cast<IVFFlatIndex*>(index_info->index_.get());
                            vector_index->InsertVectorEntry(tuple->GetValue(&child_executor_->GetOutputSchema(),0).GetVector(), *rid);
                    }
                    
                    return true;
                }
                return false;
        }
        return false;
    }
    return false;
}
}  // namespace bustub
