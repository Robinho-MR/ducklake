//===----------------------------------------------------------------------===//
//                         DuckDB
//
// storage/ducklake_deletion_snapshots.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/common/common.hpp"
#include "duckdb/common/types/value.hpp"
#include "duckdb/function/function.hpp"

namespace duckdb {
class ClientContext;
class Expression;

//! The snapshots that deleted the rows of a data file, looked up by the row number of a row in the file - the
//! bind data of the function that gives the snapshot of the rows of a deletion scan
struct DuckLakeDeletionSnapshots : public FunctionData {
public:
	//! Reads the deleted rows with their snapshot from a query that returns (row_id BIGINT, row_value BIGINT). The
	//! snapshot of a row that was not deleted by one of them is the default
	static unique_ptr<DuckLakeDeletionSnapshots> Read(ClientContext &context, const string &query,
	                                                  Value default_snapshot);
	//! The expression that gives the snapshot of the row with the number the expression returns
	static unique_ptr<Expression> CreateExpression(unique_ptr<DuckLakeDeletionSnapshots> lookup,
	                                               unique_ptr<Expression> file_row_number);

	bool IsEmpty() const {
		return rows.empty();
	}

	unique_ptr<FunctionData> Copy() const override;
	bool Equals(const FunctionData &other) const override;

public:
	//! The deleted rows, sorted, and the snapshot that deleted each of them
	vector<int64_t> rows;
	vector<int64_t> snapshots;
	Value default_snapshot;
};

} // namespace duckdb
