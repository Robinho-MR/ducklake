#include "functions/ducklake_table_functions.hpp"
#include "storage/ducklake_delete_filter.hpp"
#include "storage/ducklake_metadata_info.hpp"

#include "duckdb/common/open_file_info.hpp"
#include "duckdb/function/function_set.hpp"
#include "duckdb/function/table_function.hpp"

namespace duckdb {

//! The delete function reads a DuckLake delete file - a parquet positional delete file or a puffin deletion vector -
//! and returns the deleted positions and the snapshots that deleted them. A scan runs it in the "delete_query" of a
//! data file whose rows were deleted
struct DuckLakeReadDeleteFileData : public TableFunctionData {
	DuckLakeFileData delete_file;
};

struct DuckLakeReadDeleteFileState : public GlobalTableFunctionState {
	bool scanned = false;
	DeleteFileScanResult result;
	idx_t offset = 0;
};

static unique_ptr<FunctionData> DuckLakeReadDeleteFileBind(ClientContext &context, TableFunctionBindInput &input,
                                                           vector<LogicalType> &return_types,
                                                           vector<Identifier> &names) {
	auto &files = input.inputs[0];
	if (files.IsNull() || ListValue::GetChildren(files).size() != 1) {
		throw InvalidInputException("ducklake_read_delete_file reads exactly one delete file");
	}
	auto file = OpenFileInfo::FromValue(ListValue::GetChildren(files)[0], Identifier("ducklake_read_delete_file"));
	auto result = make_uniq<DuckLakeReadDeleteFileData>();
	result->delete_file.path = file.path;
	if (file.extended_info) {
		auto &info = *file.extended_info;
		idx_t file_size = 0;
		if (info.TryGetOption("file_size", file_size)) {
			result->delete_file.file_size_bytes = file_size;
		}
		idx_t footer_size = 0;
		if (info.TryGetOption("footer_size", footer_size)) {
			result->delete_file.footer_size = footer_size;
		}
		string encryption_key;
		if (info.TryGetOption("encryption_key", encryption_key)) {
			result->delete_file.encryption_key = std::move(encryption_key);
		}
		string format;
		if (info.TryGetOption("format", format) && format == "puffin") {
			result->delete_file.format = DeleteFileFormat::PUFFIN;
		}
	}
	names.emplace_back("pos");
	return_types.emplace_back(LogicalType::BIGINT);
	names.emplace_back("snapshot_id");
	return_types.emplace_back(LogicalType::BIGINT);
	return std::move(result);
}

static unique_ptr<GlobalTableFunctionState> DuckLakeReadDeleteFileInit(ClientContext &context,
                                                                       TableFunctionInitInput &input) {
	auto &data = input.bind_data->Cast<DuckLakeReadDeleteFileData>();
	auto result = make_uniq<DuckLakeReadDeleteFileState>();
	result->result = DuckLakeDeleteFilter::ScanDeleteFile(context, data.delete_file, optional_idx(), optional_idx());
	return std::move(result);
}

static void DuckLakeReadDeleteFileExecute(ClientContext &context, TableFunctionInput &input, DataChunk &output) {
	auto &state = input.global_state->Cast<DuckLakeReadDeleteFileState>();
	auto &rows = state.result.deleted_rows;
	auto &snapshots = state.result.snapshot_ids;
	idx_t count = MinValue<idx_t>(STANDARD_VECTOR_SIZE, rows.size() - state.offset);
	auto pos_data = FlatVector::GetDataMutable<int64_t>(output.data[0]);
	auto snapshot_data = FlatVector::GetDataMutable<int64_t>(output.data[1]);
	auto &snapshot_validity = FlatVector::ValidityMutable(output.data[1]);
	for (idx_t i = 0; i < count; i++) {
		pos_data[i] = NumericCast<int64_t>(rows[state.offset + i]);
		if (state.offset + i < snapshots.size()) {
			snapshot_data[i] = NumericCast<int64_t>(snapshots[state.offset + i]);
		} else {
			snapshot_validity.SetInvalid(i);
		}
	}
	state.offset += count;
	output.SetChildCardinality(count);
}

TableFunctionSet DuckLakeReadDeleteFileFunction::GetFunctions() {
	TableFunctionSet set("ducklake_read_delete_file");
	TableFunction function("ducklake_read_delete_file", {LogicalType::LIST(LogicalType::ANY)},
	                       DuckLakeReadDeleteFileExecute, DuckLakeReadDeleteFileBind, DuckLakeReadDeleteFileInit);
	set.AddFunction(std::move(function));
	return set;
}

} // namespace duckdb
