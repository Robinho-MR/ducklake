#include "storage/ducklake_deletion_snapshots.hpp"

#include "duckdb/common/algorithm.hpp"
#include "duckdb/common/vector/vector_iterator.hpp"
#include "duckdb/common/vector/vector_writer.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/main/connection.hpp"
#include "duckdb/main/database.hpp"
#include "duckdb/main/query_result.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"

namespace duckdb {

unique_ptr<DuckLakeDeletionSnapshots> DuckLakeDeletionSnapshots::Read(ClientContext &context, const string &query,
                                                                      Value default_snapshot) {
	// run the query on a connection of its own - this runs while the scan holds the lock of its context
	Connection connection(DatabaseInstance::GetDatabase(context));
	auto result = connection.Query(query);
	if (result->HasError()) {
		result->GetErrorObject().Throw("Failed to read the snapshots of the deleted rows: ");
	}
	vector<pair<int64_t, int64_t>> deleted;
	while (true) {
		auto chunk = result->Fetch();
		if (!chunk || chunk->size() == 0) {
			break;
		}
		auto rows = chunk->data[0].Values<int64_t>();
		auto snapshots = chunk->data[1].Values<int64_t>();
		for (idx_t i = 0; i < chunk->size(); i++) {
			if (rows[i].IsValid() && snapshots[i].IsValid()) {
				deleted.emplace_back(rows[i].GetValue(), snapshots[i].GetValue());
			}
		}
	}
	sort(deleted.begin(), deleted.end());
	auto lookup = make_uniq<DuckLakeDeletionSnapshots>();
	lookup->default_snapshot = std::move(default_snapshot);
	for (auto &entry : deleted) {
		lookup->rows.push_back(entry.first);
		lookup->snapshots.push_back(entry.second);
	}
	return lookup;
}

unique_ptr<FunctionData> DuckLakeDeletionSnapshots::Copy() const {
	auto result = make_uniq<DuckLakeDeletionSnapshots>();
	result->rows = rows;
	result->snapshots = snapshots;
	result->default_snapshot = default_snapshot;
	return std::move(result);
}

bool DuckLakeDeletionSnapshots::Equals(const FunctionData &other_p) const {
	auto &other = other_p.Cast<DuckLakeDeletionSnapshots>();
	return rows == other.rows && snapshots == other.snapshots && Value::NotDistinctFrom(default_snapshot, other.default_snapshot);
}

static void DuckLakeDeletionSnapshotFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &function_expr = state.expr.Cast<BoundFunctionExpression>();
	auto &lookup = function_expr.BindInfo()->Cast<DuckLakeDeletionSnapshots>();
	auto file_row_numbers = args.data[0].Values<int64_t>();
	auto writer = FlatVector::Writer<int64_t>(result, args.size());
	for (auto entry : file_row_numbers) {
		auto row = entry.IsValid() ? entry.GetValue() : -1;
		auto found = std::lower_bound(lookup.rows.begin(), lookup.rows.end(), row);
		if (found != lookup.rows.end() && *found == row) {
			writer.WriteValue(lookup.snapshots[NumericCast<idx_t>(found - lookup.rows.begin())]);
		} else if (!lookup.default_snapshot.IsNull()) {
			writer.WriteValue(lookup.default_snapshot.GetValue<int64_t>());
		} else {
			writer.WriteNull();
		}
	}
}

unique_ptr<Expression> DuckLakeDeletionSnapshots::CreateExpression(unique_ptr<DuckLakeDeletionSnapshots> lookup,
                                                                   unique_ptr<Expression> file_row_number) {
	ScalarFunction function("ducklake_deletion_snapshot", {LogicalType::BIGINT}, LogicalType::BIGINT,
	                        DuckLakeDeletionSnapshotFunction);
	function.SetNullHandling(FunctionNullHandling::SPECIAL_HANDLING);
	vector<unique_ptr<Expression>> children;
	children.push_back(std::move(file_row_number));
	return make_uniq<BoundFunctionExpression>(BoundScalarFunction(function), std::move(children),
	                                          cow_ptr<FunctionData>(std::move(lookup)));
}

} // namespace duckdb
