#pragma once

#include <cmath>
#include <string>
#include <utility>
#include <vector>
#include <assert.h>

#include "catalog/schema.h"
#include "execution/expressions/abstract_expression.h"
#include "fmt/format.h"
#include "storage/table/tuple.h"
#include "type/value_factory.h"

namespace bustub {

/** ComparisonType represents the type of comparison that we want to perform. */
enum class VectorExpressionType { L2Dist, InnerProduct, CosineSimilarity };

inline auto ComputeDistance(const std::vector<double> &left, const std::vector<double> &right,
                            VectorExpressionType dist_fn) {
  auto sz = left.size();
  BUSTUB_ASSERT(sz == right.size(), "vector length mismatched!");
  switch (dist_fn) {
    case VectorExpressionType::L2Dist: {
      // IMPLEMENT ME
      double distance = 0.0;
      assert(left.size() == right.size());
      for(auto i = 0; i < left.size(); i++) {
        distance += pow(left[i] - right[i], 2);
      }
      return sqrt(distance);
    }
    case VectorExpressionType::InnerProduct: {
      // IMPLEMENT ME
      double distance = 0.0;
      for(auto i = 0; i < left.size(); i++) {
        distance += left[i] * right[i];
      }
      return -1.0 * distance;
    }
    case VectorExpressionType::CosineSimilarity: {
      // IMPLEMENT ME
      double distance = 0.0;
      double left_distance = 0.0;
      double right_distance = 0.0;
      assert(left.size() == right.size());
      for(auto i = 0; i < left.size(); i++) {
        distance += left[i] * right[i];
        left_distance += pow(left[i], 2);
        right_distance += pow(right[i], 2);
      }
      return 1 - distance/sqrt(left_distance * right_distance);
    }
    default:
      BUSTUB_ASSERT(false, "Unsupported vector expr type.");
  }
}

class VectorExpression : public AbstractExpression {
 public:
  VectorExpression(VectorExpressionType expr_type, AbstractExpressionRef left, AbstractExpressionRef right)
      : AbstractExpression({std::move(left), std::move(right)}, Column{"<val>", TypeId::DECIMAL}),
        expr_type_{expr_type} {}

  auto Evaluate(const Tuple *tuple, const Schema &schema) const -> Value override {
    Value lhs = GetChildAt(0)->Evaluate(tuple, schema);
    Value rhs = GetChildAt(1)->Evaluate(tuple, schema);
    return ValueFactory::GetDecimalValue(PerformComputation(lhs, rhs));
  }

  auto EvaluateJoin(const Tuple *left_tuple, const Schema &left_schema, const Tuple *right_tuple,
                    const Schema &right_schema) const -> Value override {
    Value lhs = GetChildAt(0)->EvaluateJoin(left_tuple, left_schema, right_tuple, right_schema);
    Value rhs = GetChildAt(1)->EvaluateJoin(left_tuple, left_schema, right_tuple, right_schema);
    return ValueFactory::GetDecimalValue(PerformComputation(lhs, rhs));
  }

  /** @return the string representation of the expression node and its children */
  auto ToString() const -> std::string override {
    return fmt::format("{}({}, {})", expr_type_, *GetChildAt(0), *GetChildAt(1));
  }

  BUSTUB_EXPR_CLONE_WITH_CHILDREN(VectorExpression);

  VectorExpressionType expr_type_;

 private:
  auto PerformComputation(const Value &lhs, const Value &rhs) const -> double {
    auto left_vec = lhs.GetVector();
    auto right_vec = rhs.GetVector();
    return ComputeDistance(left_vec, right_vec, expr_type_);
  }
};

}  // namespace bustub

template <>
struct fmt::formatter<bustub::VectorExpressionType> : formatter<string_view> {
  template <typename FormatContext>
  auto format(bustub::VectorExpressionType c, FormatContext &ctx) const {
    string_view name;
    switch (c) {
      case bustub::VectorExpressionType::L2Dist:
        name = "l2_dist";
        break;
      case bustub::VectorExpressionType::CosineSimilarity:
        name = "cosine_similarity";
        break;
      case bustub::VectorExpressionType::InnerProduct:
        name = "inner_product";
        break;
      default:
        name = "Unknown";
        break;
    }
    return formatter<string_view>::format(name, ctx);
  }
};
